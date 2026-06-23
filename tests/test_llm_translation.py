import unittest
from types import SimpleNamespace

from llm_translation import (
    build_context,
    build_llm_prompt,
    build_numbered_text,
    is_llm_translation_model,
    make_translation_groups,
    parse_numbered_translations,
    render_chat_prompt,
)


class FakeTokenizer:
    chat_template = "{messages}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "CHAT:" + messages[-1]["content"]


class LlmTranslationHelpers(unittest.TestCase):
    def test_detects_qwen_instruct_as_llm(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                is_encoder_decoder=False,
                model_type="qwen2",
                architectures=["Qwen2ForCausalLM"],
            )
        )
        tokenizer = SimpleNamespace(chat_template="template")

        self.assertTrue(
            is_llm_translation_model(
                model,
                tokenizer,
                "/models/Qwen3-14B-Instruct",
            )
        )

    def test_does_not_detect_seq2seq_translation_model_as_llm(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                is_encoder_decoder=True,
                model_type="m2m_100",
                architectures=["M2M100ForConditionalGeneration"],
            )
        )
        tokenizer = SimpleNamespace(chat_template="template")

        self.assertFalse(
            is_llm_translation_model(model, tokenizer, "facebook/nllb-200-3.3B")
        )

    def test_does_not_detect_generic_causal_model_without_chat_template(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                is_encoder_decoder=False,
                model_type="gpt2",
                architectures=["GPT2LMHeadModel"],
            )
        )
        tokenizer = SimpleNamespace(chat_template=None)

        self.assertFalse(is_llm_translation_model(model, tokenizer, "gpt2"))

    def test_does_not_detect_legacy_llama_prompt_model_by_architecture_only(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                is_encoder_decoder=False,
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            )
        )
        tokenizer = SimpleNamespace(chat_template=None)

        self.assertFalse(
            is_llm_translation_model(model, tokenizer, "decapoda/llama-65b-hf")
        )

    def test_uses_chat_template_when_available(self):
        tokenizer = FakeTokenizer()
        rendered = render_chat_prompt(tokenizer, "Translate this.")

        self.assertEqual(rendered, "CHAT:Translate this.")
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.messages[0]["role"], "system")

    def test_context_formatting_excludes_current_block(self):
        lines = ["zero", "one", "two", "three", "four"]

        context = build_context(lines, start_index=2, end_index=2, window=1)

        self.assertIn("Previous block -1: one", context)
        self.assertIn("Next block +1: three", context)
        self.assertNotIn("two", context)

    def test_prompt_contains_target_context_and_text(self):
        prompt = build_llm_prompt(
            text="Original text.",
            target_language="Simplified Chinese",
            context="Previous block -1: Earlier text.",
        )

        self.assertIn("Simplified Chinese", prompt)
        self.assertIn("Previous block -1", prompt)
        self.assertIn("Original text.", prompt)

    def test_custom_prompt_requires_text_placeholder(self):
        with self.assertRaisesRegex(ValueError, "TEXT"):
            build_llm_prompt(
                text="Original text.",
                target_language="Simplified Chinese",
                prompt_template="Translate this.",
            )

    def test_merge_groups_small_neighboring_blocks(self):
        groups = make_translation_groups(
            ["short", "tiny", "this block is much longer", "mini"],
            merge_small_blocks=True,
            merge_max_chars=10,
        )

        self.assertEqual(groups, [[0, 1], [2], [3]])

    def test_numbered_block_round_trip_contract(self):
        numbered = build_numbered_text(["Alpha.", "Beta."])
        parsed = parse_numbered_translations(
            "[1] 阿尔法。\n[2] 贝塔。",
            expected_count=2,
        )

        self.assertEqual(numbered, "[1] Alpha.\n[2] Beta.")
        self.assertEqual(parsed, ["阿尔法。", "贝塔。"])

    def test_numbered_parse_rejects_missing_items(self):
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            parse_numbered_translations("[1] 阿尔法。", expected_count=2)


if __name__ == "__main__":
    unittest.main()
