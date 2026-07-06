"""Prefix cache for chat-prompt tokenization.

Multi-turn chat requests re-encode the full rendered conversation on every
turn. This cache remembers the token ids of recently encoded prompts so a new
request that shares a prefix with a cached prompt only needs to encode the new
suffix.

Correctness: BPE tokenization is not prefix-stable at arbitrary split points
(``encode(a + b) != encode(a) + encode(b)`` in general, because merges and
pre-tokenization can cross the split). The only generically safe split points
are the boundaries of *added special tokens*: HuggingFace tokenizers split the
input text on added special tokens before pre-tokenization/BPE, so no merge
ever crosses them. Chat templates delimit every message with special tokens
(e.g. ``<|im_start|>`` / ``<|im_end|>``), so a cached prefix is only ever
reused up to the end of the last special token inside the common prefix; the
few characters after it are re-encoded together with the new suffix.

Tokens with ``rstrip=True`` are excluded as cut points because they absorb
whitespace *following* the token, which would make a suffix re-encode diverge.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import OrderedDict
from typing import List, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

# Prompts shorter than this are cheap to encode; don't churn the cache.
MIN_PROMPT_CHARS = 1024

# Chunk size for the memcmp-style common-prefix scan.
_PREFIX_SCAN_CHUNK = 4096


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the common prefix of two strings, compared chunk-wise."""
    n = min(len(a), len(b))
    i = 0
    while i < n:
        j = min(i + _PREFIX_SCAN_CHUNK, n)
        if a[i:j] == b[i:j]:
            i = j
            continue
        for k in range(i, j):
            if a[k] != b[k]:
                return k
        return j
    return n


class _CacheEntry(msgspec.Struct):
    ids: List[int]
    # Parallel arrays: after the k-th special token of this prompt,
    # cut_char_ends[k] is the character position right past the special token
    # and cut_token_idx[k] is the number of tokens covering the text before it.
    cut_char_ends: List[int]
    cut_token_idx: List[int]


class TokenizerPrefixCache:
    """LRU cache of (rendered prompt -> token ids) with safe-cut metadata.

    Only called from the event loop thread (lookups happen before the encode
    await, inserts after), so no locking is needed.
    """

    def __init__(self, tokenizer, max_entries: int):
        self._max_entries = max_entries
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()

        # All added special tokens are safe split boundaries, except
        # rstrip=True ones (they absorb whitespace from the following text).
        # Tokens excluded here are simply ignored on both the string side and
        # the ids side; the pairwise verification in insert() keeps the two
        # sides consistent.
        specials = {}
        for token_id, added_token in tokenizer.added_tokens_decoder.items():
            if added_token.special and not added_token.rstrip:
                specials[token_id] = str(added_token)
        self._id_to_special = specials
        pattern = "|".join(
            re.escape(s) for s in sorted(specials.values(), key=len, reverse=True)
        )
        self._special_re = re.compile(pattern)

    def match(self, prompt: str) -> Tuple[List[int], str]:
        """Return (reusable prefix ids, remaining suffix to encode).

        On a miss, returns ``([], prompt)``.
        """
        exact = self._entries.get(prompt)
        if exact is not None:
            self._entries.move_to_end(prompt)
            return list(exact.ids), ""

        best_key = None
        best_cut_char = 0
        best_cut_token = 0
        # Most-recently-used entries first.
        for cached_prompt, entry in reversed(self._entries.items()):
            common_len = _common_prefix_len(prompt, cached_prompt)
            if common_len <= best_cut_char:
                continue
            k = bisect.bisect_right(entry.cut_char_ends, common_len) - 1
            if k < 0:
                continue
            if entry.cut_char_ends[k] > best_cut_char:
                best_key = cached_prompt
                best_cut_char = entry.cut_char_ends[k]
                best_cut_token = entry.cut_token_idx[k]

        if best_key is None:
            return [], prompt
        entry = self._entries[best_key]
        self._entries.move_to_end(best_key)
        return entry.ids[:best_cut_token], prompt[best_cut_char:]

    def insert(self, prompt: str, ids: List[int]) -> None:
        if len(prompt) < MIN_PROMPT_CHARS:
            return
        if prompt in self._entries:
            self._entries.move_to_end(prompt)
            return

        # Pair the k-th special-token occurrence in the string with the k-th
        # special id in the token list, verifying both sides agree. Any
        # mismatch (e.g. single_word tokens, exotic normalization) means we
        # cannot trust the char<->token alignment, so we simply don't cache.
        special_token_positions = [
            i for i, token_id in enumerate(ids) if token_id in self._id_to_special
        ]
        string_matches = list(self._special_re.finditer(prompt))
        if len(string_matches) != len(special_token_positions):
            logger.debug(
                "TokenizerPrefixCache: special-token count mismatch "
                "(%d in text vs %d in ids); not caching this prompt.",
                len(string_matches),
                len(special_token_positions),
            )
            return

        cut_char_ends = []
        cut_token_idx = []
        for match_, token_pos in zip(string_matches, special_token_positions):
            if self._id_to_special[ids[token_pos]] != match_.group(0):
                logger.debug(
                    "TokenizerPrefixCache: special-token order mismatch; "
                    "not caching this prompt."
                )
                return
            cut_char_ends.append(match_.end())
            cut_token_idx.append(token_pos + 1)
        if not cut_char_ends:
            return  # No safe cut point; the entry would never be reusable.

        self._entries[prompt] = _CacheEntry(
            ids=list(ids),
            cut_char_ends=cut_char_ends,
            cut_token_idx=cut_token_idx,
        )
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def create_tokenizer_prefix_cache(
    tokenizer, max_entries: int
) -> Optional[TokenizerPrefixCache]:
    """Build a TokenizerPrefixCache, or return None if unsupported.

    Requires a HuggingFace fast tokenizer (for the added-special-token split
    guarantee) with at least one usable special token.
    """
    if not getattr(tokenizer, "is_fast", False):
        # Duck-typed external tokenizers (e.g. TikToken wrappers) may not
        # define is_fast at all, hence the defensive getattr.
        logger.warning(
            "Tokenizer prefix cache disabled: requires a HuggingFace fast tokenizer."
        )
        return None
    cache = TokenizerPrefixCache(tokenizer, max_entries)
    if not cache._id_to_special:
        logger.warning(
            "Tokenizer prefix cache disabled: tokenizer has no usable "
            "special tokens to serve as safe split points."
        )
        return None
    return cache
