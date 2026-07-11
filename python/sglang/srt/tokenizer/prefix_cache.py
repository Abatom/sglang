"""Prefix cache for chat-template tokenization results.

Multi-turn conversations re-render (mostly) the same history text every turn,
and re-encoding the full rendered prompt dominates request latency at long
context lengths. This module caches ``rendered text -> token ids`` per recent
conversation and, on a lookup, re-encodes only the suffix that differs from
the cached text.

Why splicing is exact
---------------------
BPE is not prefix-stable in general (``encode(a + b) != encode(a) + encode(b)``
because merges may cross the join). However, special/added tokens are matched
by the tokenizer's added-token pre-splitter *before* the BPE model runs, so
merges never cross a special-token literal. Hence

    encode(text) == encode(text[:p]) + encode(text[p:])

holds exactly whenever ``text[p:]`` starts with a complete special-token
literal and no specials are auto-added (``add_special_tokens=False``, or the
tokenizer adds none). String-start normalizer quirks (prefix space etc.) only
affect position 0, which is always inside the cached prefix. Chat templates
delimit every message with special tokens, so cut points are plentiful.

Tokenizer families that violate the assumption are caught by an always-on
splice invariant (the first suffix token must be the cut's special id) plus an
optional verify mode that compares against a full re-encode and auto-disables
the cache after repeated mismatches. Both degrade to the full encode.

Why longest-common-prefix matching (not exact-prefix hashing): the previous
turn's full rendered text ends with the generation prompt (and the assistant
reply is re-rendered differently into the next turn's history), so it is not
an exact string prefix of the new text.

Concurrency: callers run synchronously on the tokenizer worker's event loop
(``_process_messages`` in serving_chat.py), so calls are serialized per
process and no locking is needed. With ``--tokenizer-worker-num > 1`` each
worker process holds an independent cache.
"""

from __future__ import annotations

import logging
import re
from array import array
from bisect import bisect_right
from collections import OrderedDict
from typing import Optional, Sequence

import msgspec

logger = logging.getLogger(__name__)

# A cut must save at least this many chars to be worth splicing.
MIN_USEFUL_CUT_CHARS = 4096
# On a hit covering all but this many trailing chars of the matched entry,
# replace the entry instead of inserting a sibling (typical multi-turn growth).
REPLACE_SLACK_CHARS = 16384
# First-4KB hash is the candidate index key (entries are >= min_prompt_chars).
INDEX_PREFIX_CHARS = 4096
# Block size for the chunked longest-common-prefix comparison.
LCP_CHUNK_CHARS = 256 * 1024
# Log a stats snapshot every N lookups.
STATS_LOG_INTERVAL = 1000


class SpecialBoundary(msgspec.Struct, frozen=True, kw_only=True):
    """One special-token occurrence in a cached entry (a valid cut point)."""

    char_pos: int  # offset in entry.text where the special literal starts
    token_len: int  # length of the literal in chars
    token_idx: int  # index into entry.ids of the corresponding special id
    token_id: int  # the special id itself (splice invariant check)


class PrefixCacheEntry(msgspec.Struct, kw_only=True):
    text: str  # full rendered prompt (holds a reference, no copy)
    ids: array  # array('i'), full encode result
    boundaries: list[SpecialBoundary]  # ascending char_pos
    boundary_char_ends: list[int]  # parallel [char_pos + token_len], for bisect
    add_specials_false: bool  # encode-kwargs namespace this entry was built under
    nbytes: int


class PrefixCacheStats(msgspec.Struct, kw_only=True):
    lookups: int = 0
    hits: int = 0
    chars_saved: int = 0
    splice_rejects: int = 0  # splice invariant failed at lookup
    insert_skips: int = 0  # boundary scan mismatch at insert
    verify_mismatches: int = 0
    entries: int = 0
    total_bytes: int = 0


class SpecialTokenMatcher(msgspec.Struct, kw_only=True):
    pattern: re.Pattern
    literal_to_id: dict[str, int]
    special_id_set: frozenset[int]


def build_special_matcher(*, tokenizer) -> Optional[SpecialTokenMatcher]:
    """Collect the tokenizer's special-token literals into a scanner.

    Returns None when the tokenizer does not expose the needed attributes
    (e.g. the tiktoken wrapper) or has no special tokens; the factory then
    disables the cache. The literal set is init-static: special tokens are
    fixed for the process lifetime (see utils/patch_tokenizer.py, which
    forbids adding specials after startup).

    This is the single sanctioned capability-probe point for the
    heterogeneous tokenizer object; errors are contained here.
    """
    literal_to_id: dict[str, int] = {}
    try:
        # added_tokens_decoder covers special *added* tokens that are absent
        # from all_special_tokens (e.g. Llama-3 role markers).
        for token_id, added_token in tokenizer.added_tokens_decoder.items():
            if added_token.special:
                literal_to_id[str(added_token)] = token_id
    except AttributeError:
        pass
    try:
        for literal in tokenizer.all_special_tokens:
            if literal not in literal_to_id:
                literal_to_id[literal] = tokenizer.convert_tokens_to_ids(literal)
    except AttributeError:
        pass
    literal_to_id = {
        literal: token_id
        for literal, token_id in literal_to_id.items()
        if literal and isinstance(token_id, int) and token_id >= 0
    }
    if not literal_to_id:
        return None
    # Length-descending order is load-bearing: Python `re` alternation is
    # leftmost-first, not longest-match.
    pattern = re.compile(
        "|".join(
            re.escape(literal)
            for literal in sorted(literal_to_id, key=len, reverse=True)
        )
    )
    return SpecialTokenMatcher(
        pattern=pattern,
        literal_to_id=literal_to_id,
        special_id_set=frozenset(literal_to_id.values()),
    )


def scan_boundaries(
    *,
    text: str,
    ids: Sequence[int],
    matcher: SpecialTokenMatcher,
    char_offset: int = 0,
    token_offset: int = 0,
) -> Optional[list[SpecialBoundary]]:
    """Pair special-token literals in ``text`` with special ids in ``ids``.

    They must correspond 1:1 in order (a literal special inside message
    content also encodes to the special id, so pairing still holds). Any
    count or id mismatch returns None and the caller skips caching — the
    silent-failure alternative would be a corrupt splice on the next turn.

    ``char_offset``/``token_offset`` shift the reported positions so a hit
    only needs to scan the new suffix (O(suffix) per turn).
    """
    literal_matches = [
        (m.start(), m.group()) for m in matcher.pattern.finditer(text)
    ]
    special_id_set = matcher.special_id_set
    token_indices = [i for i, t in enumerate(ids) if t in special_id_set]
    if len(literal_matches) != len(token_indices):
        return None
    boundaries = []
    literal_to_id = matcher.literal_to_id
    for (char_pos, literal), token_idx in zip(literal_matches, token_indices):
        if ids[token_idx] != literal_to_id[literal]:
            return None
        boundaries.append(
            SpecialBoundary(
                char_pos=char_offset + char_pos,
                token_len=len(literal),
                token_idx=token_offset + token_idx,
                token_id=ids[token_idx],
            )
        )
    return boundaries


def longest_common_prefix(*, a: str, b: str) -> int:
    """Length of the common prefix, via chunked C-level slice comparisons."""
    limit = min(len(a), len(b))
    pos = 0
    while pos < limit:
        step = min(LCP_CHUNK_CHARS, limit - pos)
        if a[pos : pos + step] == b[pos : pos + step]:
            pos += step
            continue
        # First differing chunk: binary-search inside it.
        lo, hi = 0, step  # invariant: a[pos:pos+lo] == b[pos:pos+lo]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[pos : pos + mid] == b[pos : pos + mid]:
                lo = mid
            else:
                hi = mid - 1
        return pos + lo
    return limit


def pick_cut(
    *, entry: PrefixCacheEntry, lcp: int
) -> Optional[SpecialBoundary]:
    """Last boundary whose special literal lies entirely inside the LCP.

    The ``char_pos + token_len <= lcp`` condition (via boundary_char_ends) is
    the correctness requirement: the new text must contain the complete
    special literal at the cut so the suffix encode starts with an atomic
    pre-split barrier.
    """
    idx = bisect_right(entry.boundary_char_ends, lcp)
    if idx == 0:
        return None
    cut = entry.boundaries[idx - 1]
    if cut.char_pos < MIN_USEFUL_CUT_CHARS:
        return None
    return cut


class TokenizerPrefixCache:
    """LRU of recent conversations' rendered text -> token ids.

    See the module docstring for the algorithm and its correctness argument.
    """

    def __init__(
        self,
        *,
        tokenizer,
        matcher: SpecialTokenMatcher,
        max_size_bytes: int,
        verify_first_n: int,
        max_verify_mismatches: int = 3,
        min_prompt_chars: int = 8192,
    ) -> None:
        self._tokenizer = tokenizer
        self._matcher = matcher
        self._max_size_bytes = max_size_bytes
        self._verify_first_n = verify_first_n
        self._max_verify_mismatches = max_verify_mismatches
        self._min_prompt_chars = min_prompt_chars
        # A tokenizer that passed build_special_matcher is HF-like and always
        # exposes bos_token_id (possibly None).
        self._bos_token_id: Optional[int] = tokenizer.bos_token_id
        self._entries: OrderedDict[int, PrefixCacheEntry] = OrderedDict()
        self._index: dict[int, list[int]] = {}  # hash(text[:4KB]) -> entry ids
        self._next_entry_id = 0
        self._total_bytes = 0
        self._stats = PrefixCacheStats()
        self._disabled = False

    def encode_with_cache(self, *, text: str, add_specials_false: bool) -> list[int]:
        if self._disabled or len(text) < self._min_prompt_chars:
            return self._full_encode(text=text, add_specials_false=add_specials_false)

        self._stats.lookups += 1
        best_id, best_lcp = self._find_best_candidate(
            text=text, add_specials_false=add_specials_false
        )
        result, cut = self._encode_via_cut(
            text=text,
            add_specials_false=add_specials_false,
            best_id=best_id,
            best_lcp=best_lcp,
        )
        self._insert(
            text=text,
            result=result,
            add_specials_false=add_specials_false,
            best_id=best_id,
            best_lcp=best_lcp,
            cut=cut,
        )
        if self._stats.lookups % STATS_LOG_INTERVAL == 0:
            self._log_stats()
        return result

    def stats(self) -> PrefixCacheStats:
        return self._stats

    # -------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------

    def _find_best_candidate(
        self, *, text: str, add_specials_false: bool
    ) -> tuple[Optional[int], int]:
        bucket = self._index.get(hash(text[:INDEX_PREFIX_CHARS]), [])
        best_id, best_lcp = None, 0
        for entry_id in bucket:
            entry = self._entries[entry_id]
            if entry.add_specials_false != add_specials_false:
                continue
            lcp = longest_common_prefix(a=entry.text, b=text)
            if lcp > best_lcp:
                best_id, best_lcp = entry_id, lcp
        return best_id, best_lcp

    def _encode_via_cut(
        self,
        *,
        text: str,
        add_specials_false: bool,
        best_id: Optional[int],
        best_lcp: int,
    ) -> tuple[list[int], Optional[SpecialBoundary]]:
        """Splice cached prefix ids with a fresh suffix encode; returns
        ``(result, cut)`` where ``cut is None`` means the full-encode path
        was taken (miss or fallback)."""
        if best_id is None:
            return (
                self._full_encode(text=text, add_specials_false=add_specials_false),
                None,
            )
        entry = self._entries[best_id]
        cut = pick_cut(entry=entry, lcp=best_lcp)
        if cut is None:
            return (
                self._full_encode(text=text, add_specials_false=add_specials_false),
                None,
            )

        suffix_ids = self._full_encode(
            text=text[cut.char_pos :], add_specials_false=add_specials_false
        )
        suffix_ids = self._strip_unexpected_bos(suffix_ids=suffix_ids, cut=cut)
        if not suffix_ids or suffix_ids[0] != cut.token_id:
            # Splice invariant failed: this tokenizer does not treat the
            # special as an atomic barrier here. Fall back to a full encode.
            self._stats.splice_rejects += 1
            return (
                self._full_encode(text=text, add_specials_false=add_specials_false),
                None,
            )

        result = entry.ids[: cut.token_idx].tolist() + suffix_ids
        self._entries.move_to_end(best_id)
        self._stats.hits += 1
        self._stats.chars_saved += cut.char_pos

        verified = self._maybe_verify(
            text=text, add_specials_false=add_specials_false, result=result
        )
        if verified is not None:
            return verified, None  # mismatch: trust the full encode
        return result, cut

    def _strip_unexpected_bos(
        self, *, suffix_ids: list[int], cut: SpecialBoundary
    ) -> list[int]:
        # Some tokenizers prepend BOS regardless of add_special_tokens
        # (mirrors _append_assistant_prefix_to_prompt_ids in serving_chat.py).
        if (
            len(suffix_ids) >= 2
            and self._bos_token_id is not None
            and suffix_ids[0] == self._bos_token_id
            and suffix_ids[0] != cut.token_id
            and suffix_ids[1] == cut.token_id
        ):
            return suffix_ids[1:]
        return suffix_ids

    def _maybe_verify(
        self, *, text: str, add_specials_false: bool, result: list[int]
    ) -> Optional[list[int]]:
        """Compare the spliced result against a full re-encode for the first
        N hits (-1 = every hit, 0 = never). Returns the full encode on
        mismatch, else None."""
        if self._verify_first_n == 0:
            return None
        if self._verify_first_n > 0 and self._stats.hits > self._verify_first_n:
            return None
        full = self._full_encode(text=text, add_specials_false=add_specials_false)
        if result == full:
            return None
        self._stats.verify_mismatches += 1
        diverge = next(
            (i for i, (a, b) in enumerate(zip(result, full)) if a != b),
            min(len(result), len(full)),
        )
        logger.warning(
            "TokenizerPrefixCache verify mismatch (%d/%d): tokenizer=%s "
            "spliced_len=%d full_len=%d first_divergence=%d. Using full encode.",
            self._stats.verify_mismatches,
            self._max_verify_mismatches,
            type(self._tokenizer).__name__,
            len(result),
            len(full),
            diverge,
        )
        if self._stats.verify_mismatches >= self._max_verify_mismatches:
            self._disabled = True
            self._entries.clear()
            self._index.clear()
            self._total_bytes = 0
            logger.warning(
                "TokenizerPrefixCache disabled after %d verify mismatches; "
                "this tokenizer does not satisfy the special-token splice "
                "assumption.",
                self._stats.verify_mismatches,
            )
        return full

    def _full_encode(self, *, text: str, add_specials_false: bool) -> list[int]:
        if add_specials_false:
            return self._tokenizer.encode(text, add_special_tokens=False)
        return self._tokenizer.encode(text)

    # -------------------------------------------------------------------
    # Insert / eviction
    # -------------------------------------------------------------------

    def _insert(
        self,
        *,
        text: str,
        result: list[int],
        add_specials_false: bool,
        best_id: Optional[int],
        best_lcp: int,
        cut: Optional[SpecialBoundary],
    ) -> None:
        if self._disabled:
            return
        boundaries = self._build_boundaries(
            text=text, result=result, best_id=best_id, cut=cut
        )
        if boundaries is None:
            self._stats.insert_skips += 1
            return
        entry = PrefixCacheEntry(
            text=text,
            ids=array("i", result),
            boundaries=boundaries,
            boundary_char_ends=[b.char_pos + b.token_len for b in boundaries],
            add_specials_false=add_specials_false,
            nbytes=len(text.encode("utf-8", errors="ignore")) + 4 * len(result),
        )
        if not boundaries:
            # Zero cut points: caching can never produce a hit. Make the
            # no-op visible instead of silently holding memory.
            logger.info(
                "TokenizerPrefixCache: prompt of %d chars contains no "
                "special-token boundaries; not caching (tokenizer=%s).",
                len(text),
                type(self._tokenizer).__name__,
            )
            return
        # Typical multi-turn growth: the new text subsumes the matched entry;
        # replace it so each conversation keeps one entry. A short LCP means
        # a sibling conversation (e.g. shared system prompt): insert anew.
        if (
            best_id is not None
            and best_lcp >= len(self._entries[best_id].text) - REPLACE_SLACK_CHARS
        ):
            self._remove_entry(best_id)
        self._add_entry(entry)
        self._evict_to_budget()

    def _build_boundaries(
        self,
        *,
        text: str,
        result: list[int],
        best_id: Optional[int],
        cut: Optional[SpecialBoundary],
    ) -> Optional[list[SpecialBoundary]]:
        if cut is not None and best_id is not None:
            # Hit: text[:cut.char_pos] is byte-identical to the matched
            # entry's prefix, so its boundaries carry over; scan only the
            # suffix. result[cut.token_idx:] is exactly the (BOS-stripped)
            # suffix encode by construction.
            prefix_boundaries = [
                b
                for b in self._entries[best_id].boundaries
                if b.char_pos + b.token_len <= cut.char_pos
            ]
            suffix_boundaries = scan_boundaries(
                text=text[cut.char_pos :],
                ids=result[cut.token_idx :],
                matcher=self._matcher,
                char_offset=cut.char_pos,
                token_offset=cut.token_idx,
            )
            if suffix_boundaries is None:
                return None
            return prefix_boundaries + suffix_boundaries
        return scan_boundaries(text=text, ids=result, matcher=self._matcher)

    def _add_entry(self, entry: PrefixCacheEntry) -> None:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        self._entries[entry_id] = entry
        self._index.setdefault(hash(entry.text[:INDEX_PREFIX_CHARS]), []).append(
            entry_id
        )
        self._total_bytes += entry.nbytes
        self._stats.entries = len(self._entries)
        self._stats.total_bytes = self._total_bytes

    def _remove_entry(self, entry_id: int) -> None:
        entry = self._entries.pop(entry_id)
        key = hash(entry.text[:INDEX_PREFIX_CHARS])
        bucket = self._index[key]
        bucket.remove(entry_id)
        if not bucket:
            del self._index[key]
        self._total_bytes -= entry.nbytes
        self._stats.entries = len(self._entries)
        self._stats.total_bytes = self._total_bytes

    def _evict_to_budget(self) -> None:
        while self._total_bytes > self._max_size_bytes and len(self._entries) > 1:
            oldest_id = next(iter(self._entries))
            self._remove_entry(oldest_id)

    def _log_stats(self) -> None:
        s = self._stats
        logger.info(
            "TokenizerPrefixCache: hits=%d/%d (%.1f%%), saved=%.3g chars, "
            "entries=%d, %.1f/%.1f MB",
            s.hits,
            s.lookups,
            100.0 * s.hits / max(s.lookups, 1),
            float(s.chars_saved),
            s.entries,
            s.total_bytes / 1e6,
            self._max_size_bytes / 1e6,
        )


def maybe_create_tokenizer_prefix_cache(
    *,
    tokenizer,
    enable: bool,
    max_size_mb: int,
    verify_first_n: int,
) -> Optional[TokenizerPrefixCache]:
    """Factory: returns None (cache disabled) unless the flag is on and the
    tokenizer exposes enumerable special tokens."""
    if not enable:
        return None
    if tokenizer is None:
        logger.warning(
            "--enable-tokenizer-prefix-cache has no effect without a tokenizer."
        )
        return None
    matcher = build_special_matcher(tokenizer=tokenizer)
    if matcher is None:
        logger.warning(
            "--enable-tokenizer-prefix-cache disabled: tokenizer %s does not "
            "expose special tokens (all_special_tokens / added_tokens_decoder).",
            type(tokenizer).__name__,
        )
        return None
    return TokenizerPrefixCache(
        tokenizer=tokenizer,
        matcher=matcher,
        max_size_bytes=max_size_mb * 1024 * 1024,
        verify_first_n=verify_first_n,
    )
