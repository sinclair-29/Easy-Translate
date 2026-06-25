import unittest
from types import SimpleNamespace

from llm_translation import (
    apply_chat_template_tokenized,
    apply_translategemma_chat_template_tokenized,
    build_context,
    build_llm_prompt,
    build_numbered_text,
    build_translategemma_messages,
    is_llm_translation_model,
    mark_translategemma_processor,
    make_translation_groups,
    parse_numbered_translations,
    render_chat_prompt,
    resolve_translategemma_language_codes,
    split_text_for_llm,
    translategemma_language_code_candidates,
)


class FakeTokenizer:
    chat_template = "{messages}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "CHAT:" + messages[-1]["content"]


class FakeTokenizedTokenizer:
    chat_template = "{messages}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors, enable_thinking=False):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.return_tensors = return_tensors
        self.enable_thinking = enable_thinking
        return [[1, 2, 3]]


class FakeTranslateGemmaProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.return_dict = return_dict
        self.return_tensors = return_tensors
        return {"input_ids": [[4, 5, 6]]}


class RestrictedTranslateGemmaProcessor(FakeTranslateGemmaProcessor):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        content = messages[0]["content"][0]
        if content["target_lang_code"] != "zh":
            raise KeyError(content["target_lang_code"])
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors=return_tensors,
        )


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

    def test_detects_translategemma_processor(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                is_encoder_decoder=False,
                model_type="gemma3",
                architectures=["Gemma3ForConditionalGeneration"],
            )
        )
        processor = mark_translategemma_processor(FakeTranslateGemmaProcessor())

        self.assertTrue(
            is_llm_translation_model(
                model,
                processor,
                "google/translategemma-12b-it",
            )
        )

    def test_translategemma_message_contains_language_codes(self):
        messages = build_translategemma_messages(
            text="Hello world.",
            source_lang_code="en",
            target_lang_code="zh-CN",
        )

        content = messages[0]["content"][0]
        self.assertEqual(content["type"], "text")
        self.assertEqual(content["source_lang_code"], "en")
        self.assertEqual(content["target_lang_code"], "zh-CN")
        self.assertEqual(content["text"], "Hello world.")

    def test_translategemma_tokenization_uses_processor_template(self):
        processor = mark_translategemma_processor(FakeTranslateGemmaProcessor())

        tokenized = apply_translategemma_chat_template_tokenized(
            processor,
            "Hello world.",
            source_lang_code="en",
            target_lang_code="zh-CN",
        )

        self.assertEqual(tokenized["input_ids"], [[4, 5, 6]])
        self.assertTrue(processor.tokenize)
        self.assertTrue(processor.add_generation_prompt)
        self.assertTrue(processor.return_dict)
        self.assertEqual(processor.return_tensors, "pt")

    def test_translategemma_language_code_candidates_include_base_language(self):
        self.assertIn("zh", translategemma_language_code_candidates("zh-CN"))

    def test_resolves_translategemma_language_code_alias(self):
        processor = mark_translategemma_processor(RestrictedTranslateGemmaProcessor())

        source_code, target_code = resolve_translategemma_language_codes(
            processor,
            source_lang_code="en",
            target_lang_code="zh-CN",
        )

        self.assertEqual(source_code, "en")
        self.assertEqual(target_code, "zh")

    def test_uses_chat_template_when_available(self):
        tokenizer = FakeTokenizer()
        rendered = render_chat_prompt(tokenizer, "Translate this.")

        self.assertEqual(rendered, "CHAT:Translate this.")
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.messages[0]["role"], "system")

    def test_tokenized_chat_template_disables_thinking_when_supported(self):
        tokenizer = FakeTokenizedTokenizer()
        tokenized = apply_chat_template_tokenized(tokenizer, "Translate this.")

        self.assertEqual(tokenized, [[1, 2, 3]])
        self.assertTrue(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.return_tensors, "pt")
        self.assertFalse(tokenizer.enable_thinking)

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

    def test_split_text_for_llm_prefers_sentence_boundaries(self):
        chunks = split_text_for_llm(
            "First sentence. Second sentence. Third sentence.",
            max_chars=30,
        )

        self.assertEqual(chunks, ["First sentence.", "Second sentence.", "Third sentence."])

    def test_split_text_for_llm_keeps_short_text_unchanged(self):
        self.assertEqual(split_text_for_llm("short text", max_chars=30), ["short text"])


if __name__ == "__main__":
    unittest.main()
