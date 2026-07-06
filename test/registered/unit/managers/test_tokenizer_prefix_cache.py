"""Unit tests for srt/managers/tokenizer_prefix_cache.py"""

import re
import unittest

from sglang.srt.managers.tokenizer_prefix_cache import (
    MIN_PROMPT_CHARS,
    TokenizerPrefixCache,
    create_tokenizer_prefix_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IM_START = "<|s|>"
IM_END = "<|e|>"
IM_START_ID = 1000
IM_END_ID = 1001


class _FakeAddedToken:
    def __init__(self, content, special=True, rstrip=False):
        self.content = content
        self.special = special
        self.rstrip = rstrip
        self.lstrip = False

    def __str__(self):
        return self.content


class _FakeFastTokenizer:
    """Char-level tokenizer with special tokens.

    Splits on special tokens first (like HF fast tokenizers), then encodes
    one token per character, so ``encode(a) + encode(b) == encode(a + b)``
    holds exactly at special-token boundaries — letting the tests verify the
    cache's correctness invariant precisely.
    """

    is_fast = True

    def __init__(self, specials=None):
        self._specials = specials or {
            IM_START_ID: _FakeAddedToken(IM_START),
            IM_END_ID: _FakeAddedToken(IM_END),
        }
        strings = [str(t) for t in self._specials.values()]
        self._re = re.compile("|".join(re.escape(s) for s in strings))
        self._rev = {str(t): tid for tid, t in self._specials.items()}

    @property
    def added_tokens_decoder(self):
        return self._specials

    def encode(self, text, **kwargs):
        ids = []
        pos = 0
        for m in self._re.finditer(text):
            ids.extend(ord(c) for c in text[pos : m.start()])
            ids.append(self._rev[m.group(0)])
            pos = m.end()
        ids.extend(ord(c) for c in text[pos:])
        return ids


def _render(messages, add_generation_prompt=True):
    """Mimic a chat template: <|s|>role\\ncontent<|e|>\\n per message."""
    parts = [f"{IM_START}{role}\n{content}{IM_END}\n" for role, content in messages]
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


LONG_CONTENT = "x" * (MIN_PROMPT_CHARS + 200)  # ensure prompts are cacheable


class TestTokenizerPrefixCache(CustomTestCase):
    def setUp(self):
        self.tokenizer = _FakeFastTokenizer()
        self.cache = TokenizerPrefixCache(self.tokenizer, max_entries=4)

    def _insert(self, prompt):
        ids = self.tokenizer.encode(prompt)
        self.cache.insert(prompt, ids)
        return ids

    def test_multi_turn_hit_matches_full_encode(self):
        turn1 = _render([("system", "sys"), ("user", LONG_CONTENT)])
        self._insert(turn1)

        turn2 = _render(
            [
                ("system", "sys"),
                ("user", LONG_CONTENT),
                ("assistant", "the reply"),
                ("user", "next question"),
            ]
        )
        prefix_ids, suffix = self.cache.match(turn2)

        # The cut is at the trailing <|s|> of turn1's generation prompt:
        # only "assistant\n" plus the new messages are re-encoded.
        self.assertTrue(prefix_ids)
        self.assertTrue(suffix.startswith("assistant\n"))
        self.assertLess(len(suffix), len(turn2) - len(turn1) + len("assistant\n") + 1)

        # Correctness invariant: reused prefix + suffix encode == full encode.
        self.assertEqual(
            prefix_ids + self.tokenizer.encode(suffix),
            self.tokenizer.encode(turn2),
        )

    def test_exact_hit_returns_full_ids(self):
        prompt = _render([("user", LONG_CONTENT)])
        ids = self._insert(prompt)
        prefix_ids, suffix = self.cache.match(prompt)
        self.assertEqual(prefix_ids, ids)
        self.assertEqual(suffix, "")

    def test_miss_on_unrelated_prompt(self):
        self._insert(_render([("user", LONG_CONTENT)]))
        other = "y" * 2000
        prefix_ids, suffix = self.cache.match(other)
        self.assertEqual(prefix_ids, [])
        self.assertEqual(suffix, other)

    def test_miss_when_no_special_token_inside_common_prefix(self):
        # Shares only part of the first message content: the common prefix
        # ends before any special-token *end* boundary is usable... the first
        # boundary is the end of <|s|>, which IS within the common prefix, so
        # construct prompts diverging inside the special token region instead.
        cached = _render([("user", LONG_CONTENT)])
        self._insert(cached)
        # Diverges inside the very first special token.
        new = "<|different" + "z" * 2000
        prefix_ids, suffix = self.cache.match(new)
        self.assertEqual(prefix_ids, [])
        self.assertEqual(suffix, new)

    def test_partial_hit_cuts_to_last_special_in_common_prefix(self):
        cached = _render([("system", "sys"), ("user", LONG_CONTENT + "AAAA")])
        self._insert(cached)
        # Same system message, user content diverges mid-body.
        new = _render([("system", "sys"), ("user", LONG_CONTENT + "BBBB")])
        prefix_ids, suffix = self.cache.match(new)
        # Reusable region ends at the <|s|> that opens the user message.
        self.assertTrue(prefix_ids)
        self.assertTrue(suffix.startswith("user\n"))
        self.assertEqual(
            prefix_ids + self.tokenizer.encode(suffix),
            self.tokenizer.encode(new),
        )

    def test_lru_eviction(self):
        cache = TokenizerPrefixCache(self.tokenizer, max_entries=2)
        prompts = [
            _render([("user", LONG_CONTENT + str(i))]) for i in range(3)
        ]
        for p in prompts:
            cache.insert(p, self.tokenizer.encode(p))
        # Oldest entry evicted.
        prefix_ids, suffix = cache.match(prompts[0])
        # prompts share a long common prefix, so a *partial* hit against the
        # surviving entries is expected — but not an exact one.
        self.assertNotEqual((prefix_ids, suffix), (self.tokenizer.encode(prompts[0]), ""))
        # Newest entries still give exact hits.
        for p in prompts[1:]:
            ids, s = cache.match(p)
            self.assertEqual(s, "")
            self.assertEqual(ids, self.tokenizer.encode(p))

    def test_short_prompts_not_cached(self):
        prompt = _render([("user", "short")])
        self.assertLess(len(prompt), MIN_PROMPT_CHARS)
        self.cache.insert(prompt, self.tokenizer.encode(prompt))
        prefix_ids, suffix = self.cache.match(prompt)
        self.assertEqual(prefix_ids, [])
        self.assertEqual(suffix, prompt)

    def test_special_count_mismatch_not_cached(self):
        prompt = _render([("user", LONG_CONTENT)])
        ids = self.tokenizer.encode(prompt)
        # Corrupt ids: drop one special token id.
        bad_ids = [t for t in ids if t != IM_END_ID]
        self.cache.insert(prompt, bad_ids)
        prefix_ids, suffix = self.cache.match(prompt)
        self.assertEqual(prefix_ids, [])
        self.assertEqual(suffix, prompt)

    def test_returned_prefix_ids_are_not_aliased(self):
        prompt = _render([("user", LONG_CONTENT)])
        ids = self._insert(prompt)
        got, _ = self.cache.match(prompt)
        got.append(-1)
        again, _ = self.cache.match(prompt)
        self.assertEqual(again, ids)


class TestCreateTokenizerPrefixCache(CustomTestCase):
    def test_fast_tokenizer_with_specials_supported(self):
        cache = create_tokenizer_prefix_cache(_FakeFastTokenizer(), 8)
        self.assertIsInstance(cache, TokenizerPrefixCache)

    def test_non_fast_tokenizer_unsupported(self):
        tokenizer = _FakeFastTokenizer()
        tokenizer.is_fast = False
        self.assertIsNone(create_tokenizer_prefix_cache(tokenizer, 8))

    def test_rstrip_only_specials_unsupported(self):
        specials = {
            IM_START_ID: _FakeAddedToken(IM_START, rstrip=True),
        }
        tokenizer = _FakeFastTokenizer(specials={**specials})
        self.assertIsNone(create_tokenizer_prefix_cache(tokenizer, 8))

    def test_non_special_added_tokens_ignored(self):
        specials = {
            IM_START_ID: _FakeAddedToken(IM_START, special=False),
        }
        tokenizer = _FakeFastTokenizer(specials={**specials})
        self.assertIsNone(create_tokenizer_prefix_cache(tokenizer, 8))


if __name__ == "__main__":
    unittest.main()
