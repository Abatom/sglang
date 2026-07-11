"""Unit tests for srt/tokenizer/prefix_cache.

The cache splices ``cached_prefix_ids + encode(suffix)`` and is only correct
because special tokens are atomic pre-split barriers; these tests pin down
that derivation and the negative branches that keep a bad splice from ever
reaching a request (scan mismatch -> no insert, invariant/verify failure ->
full encode).
"""

import unittest

from sglang.srt.tokenizer.prefix_cache import (
    SpecialTokenMatcher,
    TokenizerPrefixCache,
    build_special_matcher,
    longest_common_prefix,
    maybe_create_tokenizer_prefix_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

QWEN_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Stub tokenizer: "<S>" is the only special token (id 0); every other char
# encodes position-independently to one id. Splicing is exact by construction,
# which lets the tests target the cache's bookkeeping in isolation.
# ---------------------------------------------------------------------------

SPECIAL = "<S>"
SPECIAL_ID = 0


class _StubTokenizer:
    bos_token_id = None

    def encode(self, text, **kwargs):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(SPECIAL, i):
                ids.append(SPECIAL_ID)
                i += len(SPECIAL)
            else:
                ids.append(1 + ord(text[i]) % 100)
                i += 1
        return ids


def _stub_matcher() -> SpecialTokenMatcher:
    import re

    return SpecialTokenMatcher(
        pattern=re.compile(re.escape(SPECIAL)),
        literal_to_id={SPECIAL: SPECIAL_ID},
        special_id_set=frozenset({SPECIAL_ID}),
    )


def _make_cache(tokenizer, **kwargs) -> TokenizerPrefixCache:
    defaults = dict(
        matcher=_stub_matcher(),
        max_size_bytes=64 * 1024 * 1024,
        verify_first_n=0,
        min_prompt_chars=1,
    )
    defaults.update(kwargs)
    return TokenizerPrefixCache(tokenizer=tokenizer, **defaults)


def _stub_turn_texts(num_turns: int) -> list[str]:
    """Conversation-shaped texts: a long prefix then <S>-delimited turns."""
    base = "system " * 1000  # ~7000 chars so cuts clear MIN_USEFUL_CUT_CHARS
    texts = []
    text = base
    for turn in range(num_turns):
        text += f"{SPECIAL}user says thing number {turn} {'x' * 200}{SPECIAL}"
        texts.append(text + f"{SPECIAL}assistant:")  # generation-prompt tail
        text += f"{SPECIAL}assistant reply {turn} {'y' * 100}"
    return texts


class TestPrefixCacheWithStub(CustomTestCase):
    def test_spliced_equals_full_encode_multi_turn_stub(self):
        """Derived property: splice exactness over multi-turn growth.

        Red if the LCP math, cut selection, or incremental boundary scan
        regresses -- any of those silently yields wrong token ids."""
        tok = _StubTokenizer()
        cache = _make_cache(tok)
        for text in _stub_turn_texts(6):
            got = cache.encode_with_cache(text=text, add_specials_false=False)
            self.assertEqual(got, tok.encode(text))
        stats = cache.stats()
        self.assertGreaterEqual(stats.hits, 4)
        # replace-on-hit keeps one entry for a single growing conversation
        self.assertEqual(stats.entries, 1)

    def test_cut_never_lands_inside_special(self):
        """Derived property: the cut must satisfy char_pos + token_len <= LCP.

        Red if that degrades to char_pos <= LCP: a divergence in the middle
        of a special literal would splice a truncated special."""
        tok = _StubTokenizer()
        cache = _make_cache(tok)
        first = _stub_turn_texts(3)[-1]
        cache.encode_with_cache(text=first, add_specials_false=False)
        # Diverge one char into the *last* special literal occurrence.
        pos = first.rindex(SPECIAL)
        diverged = first[: pos + 1] + "!DIVERGED!" + "z" * 200
        got = cache.encode_with_cache(text=diverged, add_specials_false=False)
        self.assertEqual(got, tok.encode(diverged))
        stats = cache.stats()
        self.assertEqual(stats.splice_rejects, 0)
        if stats.hits:  # if it spliced, the cut must predate the broken special
            self.assertLessEqual(stats.chars_saved, pos)

    def test_boundary_scan_mismatch_skips_insert(self):
        """Negative-branch contract: literal/id count mismatch must skip the
        insert. Red if the pairing check is dropped -- the silent-failure
        alternative is a corrupt splice on the next turn."""

        class _NoSpecialIdsTokenizer(_StubTokenizer):
            def encode(self, text, **kwargs):
                return [7] * (len(text) // 4)  # never emits SPECIAL_ID

        tok = _NoSpecialIdsTokenizer()
        cache = _make_cache(tok)
        text = _stub_turn_texts(1)[0]
        got = cache.encode_with_cache(text=text, add_specials_false=False)
        self.assertEqual(got, tok.encode(text))
        stats = cache.stats()
        self.assertEqual(stats.entries, 0)
        self.assertEqual(stats.insert_skips, 1)

    def test_lru_eviction_by_bytes(self):
        """Bookkeeping: byte-budget eviction, index cleanup, byte conservation.

        Red if eviction forgets the index bucket or drifts total_bytes."""
        tok = _StubTokenizer()
        texts = [
            f"conversation {i} " * 500 + SPECIAL + "tail " * 1200 + SPECIAL
            for i in range(3)
        ]
        entry_bytes = len(texts[0].encode()) + 4 * len(tok.encode(texts[0]))
        cache = _make_cache(tok, max_size_bytes=int(entry_bytes * 2.5))
        for text in texts:
            cache.encode_with_cache(text=text, add_specials_false=False)
        stats = cache.stats()
        self.assertEqual(stats.entries, 2)  # oldest evicted
        remaining = list(cache._entries.values())
        self.assertEqual({e.text for e in remaining}, set(texts[1:]))
        self.assertEqual(stats.total_bytes, sum(e.nbytes for e in remaining))
        indexed_ids = [i for bucket in cache._index.values() for i in bucket]
        self.assertEqual(sorted(indexed_ids), sorted(cache._entries.keys()))

    def test_verify_mode_fallback_and_autodisable(self):
        """Negative branch: on a verify mismatch the *full* encode must win,
        and repeated mismatches must disable the cache entirely. Red if
        verify compares-but-returns-spliced or auto-disable is lost."""

        class _InconsistentTokenizer(_StubTokenizer):
            # Suffix encodes (text starting with the special) shift all
            # non-special ids by +1: splices disagree with full encodes while
            # the splice invariant (first token == special id) still passes.
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                if text.startswith(SPECIAL):
                    ids = [t if t == SPECIAL_ID else t + 1 for t in ids]
                return ids

        tok = _InconsistentTokenizer()
        cache = _make_cache(tok, verify_first_n=-1, max_verify_mismatches=2)
        texts = _stub_turn_texts(6)
        for text in texts:
            got = cache.encode_with_cache(text=text, add_specials_false=False)
            self.assertEqual(got, tok.encode(text))  # full encode always wins
        stats = cache.stats()
        self.assertEqual(stats.verify_mismatches, 2)
        self.assertEqual(stats.entries, 0)  # disabled cache is emptied
        lookups_when_disabled = stats.lookups
        cache.encode_with_cache(text=texts[-1], add_specials_false=False)
        self.assertEqual(stats.lookups, lookups_when_disabled)  # full bypass

    def test_bos_stripped_from_suffix_encode(self):
        """Derived property mirroring _append_assistant_prefix_to_prompt_ids:
        a tokenizer that force-prepends BOS must not corrupt the splice."""
        BOS = 99

        class _BosTokenizer(_StubTokenizer):
            bos_token_id = BOS

            def encode(self, text, **kwargs):
                return [BOS] + super().encode(text, **kwargs)

        tok = _BosTokenizer()
        cache = _make_cache(tok, matcher=_stub_matcher())
        texts = _stub_turn_texts(3)
        for text in texts:
            got = cache.encode_with_cache(text=text, add_specials_false=False)
            self.assertEqual(got, tok.encode(text))
        self.assertGreaterEqual(cache.stats().hits, 1)

    def test_non_bos_junk_prefix_rejects_splice(self):
        """Negative branch: an unexpected leading id that is NOT the BOS must
        fail the splice invariant and fall back to the full encode. Red if
        the invariant degrades to always-true."""
        JUNK = 98

        class _JunkTokenizer(_StubTokenizer):
            bos_token_id = None

            def encode(self, text, **kwargs):
                return [JUNK] + super().encode(text, **kwargs)

        tok = _JunkTokenizer()
        cache = _make_cache(tok)
        for text in _stub_turn_texts(3):
            got = cache.encode_with_cache(text=text, add_specials_false=False)
            self.assertEqual(got, tok.encode(text))
        stats = cache.stats()
        self.assertEqual(stats.hits, 0)
        self.assertGreaterEqual(stats.splice_rejects, 1)

    def test_encode_kwargs_namespaces_do_not_mix(self):
        """Negative branch: entries built under add_special_tokens=False must
        not serve lookups from the other kwargs namespace (their ids differ
        on real tokenizers). Red if the namespace filter is dropped."""
        tok = _StubTokenizer()
        cache = _make_cache(tok)
        text = _stub_turn_texts(1)[0]
        cache.encode_with_cache(text=text, add_specials_false=True)
        cache.encode_with_cache(text=text, add_specials_false=False)
        stats = cache.stats()
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.entries, 2)

    def test_short_prompt_bypasses_cache(self):
        """Default-applied contract: prompts below min_prompt_chars must not
        touch the cache (no lookup, no insert) -- churn guard."""
        tok = _StubTokenizer()
        cache = _make_cache(tok, min_prompt_chars=8192)
        short = SPECIAL + "hi" * 100 + SPECIAL
        got = cache.encode_with_cache(text=short, add_specials_false=False)
        self.assertEqual(got, tok.encode(short))
        stats = cache.stats()
        self.assertEqual(stats.lookups, 0)
        self.assertEqual(stats.entries, 0)

    def test_factory_disables_without_special_tokens(self):
        """Negative branch: a tokenizer exposing no special tokens (e.g. the
        tiktoken wrapper) must yield None, not a never-hitting cache."""

        class _Bare:
            pass

        self.assertIsNone(
            maybe_create_tokenizer_prefix_cache(
                tokenizer=_Bare(), enable=True, max_size_mb=1, verify_first_n=0
            )
        )
        self.assertIsNone(
            maybe_create_tokenizer_prefix_cache(
                tokenizer=None, enable=True, max_size_mb=1, verify_first_n=0
            )
        )

    def test_longest_common_prefix_boundaries(self):
        """Derived property: chunked LCP must be exact at chunk edges."""
        from sglang.srt.tokenizer import prefix_cache as pc

        chunk = pc.LCP_CHUNK_CHARS
        a = "a" * (chunk * 2 + 17)
        for diff_at in (0, 1, chunk - 1, chunk, chunk + 1, len(a) - 1):
            b = a[:diff_at] + "b" + a[diff_at + 1 :]
            self.assertEqual(longest_common_prefix(a=a, b=b), diff_at)
        self.assertEqual(longest_common_prefix(a=a, b=a), len(a))
        self.assertEqual(longest_common_prefix(a=a, b=a[: chunk + 3]), chunk + 3)


class TestPrefixCacheWithRealTokenizer(CustomTestCase):
    """End-to-end splice exactness against a real HF fast tokenizer and its
    chat template (special tokens delimit every message)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)

    def _render(self, messages) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _make_real_cache(self, **kwargs) -> TokenizerPrefixCache:
        matcher = build_special_matcher(tokenizer=self.tokenizer)
        self.assertIsNotNone(matcher)
        defaults = dict(
            matcher=matcher,
            max_size_bytes=64 * 1024 * 1024,
            verify_first_n=0,
            min_prompt_chars=1,
        )
        defaults.update(kwargs)
        return TokenizerPrefixCache(tokenizer=self.tokenizer, **defaults)

    def test_spliced_equals_full_encode_multi_turn(self):
        """Derived property (the whole point of the module): across a growing
        conversation, spliced ids must equal a full re-encode, including when
        message content embeds a literal special token. Red if the atomic-
        barrier assumption is misapplied anywhere in the pipeline."""
        cache = self._make_real_cache()
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 200}
        ]
        for turn in range(8):
            content = f"Question {turn}: " + f"details {turn} " * 60
            if turn == 4:
                # literal special token inside content
                content += " sneaky <|im_end|> embedded <|im_start|> literal "
            messages.append({"role": "user", "content": content})
            rendered = self._render(messages)
            got = cache.encode_with_cache(text=rendered, add_specials_false=True)
            expected = self.tokenizer.encode(rendered, add_special_tokens=False)
            self.assertEqual(got, expected, f"divergence at turn {turn}")
            messages.append(
                {"role": "assistant", "content": f"Answer {turn}. " * 40}
            )
        stats = cache.stats()
        self.assertGreaterEqual(stats.hits, 5)
        self.assertEqual(stats.splice_rejects, 0)
        self.assertEqual(stats.insert_skips, 0)

    def test_verify_mode_passes_on_real_tokenizer(self):
        """Always-verify over real multi-turn growth: zero mismatches. Red if
        the splice is subtly wrong in a way the equality test above happens
        not to cover (kwargs drift between full and suffix encodes, etc.)."""
        cache = self._make_real_cache(verify_first_n=-1)
        messages = [{"role": "system", "content": "Be concise. " * 400}]
        for turn in range(4):
            messages.append({"role": "user", "content": f"q{turn} " * 100})
            rendered = self._render(messages)
            cache.encode_with_cache(text=rendered, add_specials_false=True)
            messages.append({"role": "assistant", "content": f"a{turn} " * 50})
        stats = cache.stats()
        self.assertGreaterEqual(stats.hits, 2)
        self.assertEqual(stats.verify_mismatches, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
