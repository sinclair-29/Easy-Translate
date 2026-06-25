from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from typing import Optional, Tuple

import torch

import json

from llm_translation import is_translategemma_name, mark_translategemma_processor


UNSUPPORTED_SEQ2SEQ_MODEL_TYPES = {
    "fsmt",
    "m2m_100",
    "marian",
    "mbart",
    "mt5",
    "nllb",
    "seamless_m4t",
    "t5",
}

LLM_ONLY_MODEL_ERROR = (
    "This fork/version of Easy-Translate is now LLM-only and expects an "
    "instruction-tuned local translation model such as Qwen3-14B-Instruct or "
    "google/translategemma-12b-it. "
    "Legacy Seq2Seq translation models such as NLLB, M2M100, SeamlessM4T, "
    "MBART, MarianMT, and T5 are no longer supported."
)


def load_model_for_inference(
    weights_path: str,
    quantization: Optional[int] = None,
    lora_weights_name_or_path: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    force_auto_device_map: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load an instruction-tuned causal language model for inference.

    Args:
        weights_path (`str`):
            The path to your local model weights and tokenizer. You can also provide a
            huggingface hub model name.
        quantization (`int`, optional):
            '4' or '8' for 4 bits or 8 bits quantization or None for 16/32bits training. Defaults to `None`.

            Requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes
        lora_weights_name_or_path (`Optional[str]`, optional):
            If the model has been trained with LoRA, path or huggingface hub name to the
            pretrained weights. Defaults to `None`.
        torch_dtype (`Optional[str]`, optional):
            The torch dtype to use for the model. If set to `"auto"`, the dtype will be
            automatically derived. Defaults to `None`. If quantization is enabled, we will override
            this to 'torch.bfloat16'.
        force_auto_device_map (`bool`, optional):
            Whether to force the use of the auto device map. If set to True, the model will be split across
            GPUs and CPU to fit the model in memory. If set to False, a full copy of the model will be loaded
            into each GPU. Defaults to False.
        trust_remote_code (`bool`, optional):
            Trust the remote code from HuggingFace model hub. Defaults to False.
        attn_implementation (`Optional[str]`, optional):
            HuggingFace attention implementation to use, such as
            "flash_attention_2" or "sdpa". Defaults to `None`, which keeps the
            Transformers default behavior.

    Returns:
        `Tuple[PreTrainedModel, PreTrainedTokenizerBase]`:
            The loaded model and tokenizer.
    """

    if isinstance(quantization, str):
        quantization = int(quantization)
    assert (quantization is None) or (
        quantization in [4, 8]
    ), f"Quantization must be 4 or 8, or None for FP32/FP16 training. You passed: {quantization}"

    print(f"Loading model from {weights_path}")

    config = AutoConfig.from_pretrained(
        weights_path, trust_remote_code=trust_remote_code
    )

    torch_dtype = (
        torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)
    )

    model_type = str(getattr(config, "model_type", "")).lower()
    is_translategemma = is_translategemma_name(weights_path)
    if getattr(config, "is_encoder_decoder", False) or model_type in UNSUPPORTED_SEQ2SEQ_MODEL_TYPES:
        raise ValueError(
            f"Model {weights_path} has model_type={config.model_type!r}. "
            f"{LLM_ONLY_MODEL_ERROR}"
        )

    quant_args = {}

    if quantization is not None:
        quant_args = (
            {"load_in_4bit": True} if quantization == 4 else {"load_in_8bit": True}
        )
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch_dtype in ["auto", None]
                else torch_dtype,
            )

        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        print(
            f"Bits and Bytes config: {json.dumps(bnb_config.to_dict(), indent=4, ensure_ascii=False)}"
        )
    else:
        print(f"Loading model with dtype: {torch_dtype}")
        bnb_config = None

    if is_translategemma:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise ImportError(
                "TranslateGemma requires a recent Transformers version with "
                "AutoProcessor and AutoModelForImageTextToText. Upgrade with: "
                "pip install --upgrade transformers"
            ) from error

        processor = AutoProcessor.from_pretrained(
            weights_path,
            trust_remote_code=trust_remote_code,
        )
        processor = mark_translategemma_processor(processor)
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"

        print(
            f"Model {weights_path} is a TranslateGemma image-text translation model. "
            "We will load it with AutoModelForImageTextToText."
        )
        model_kwargs = {
            "pretrained_model_name_or_path": weights_path,
            "device_map": "auto" if force_auto_device_map else None,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "quantization_config": bnb_config,
            **quant_args,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model: PreTrainedModel = AutoModelForImageTextToText.from_pretrained(
            **model_kwargs
        )

        if lora_weights_name_or_path:
            from peft import PeftModel

            print(f"Loading pretrained LORA weights from {lora_weights_name_or_path}")
            model = PeftModel.from_pretrained(model, lora_weights_name_or_path)
            if quantization is None:
                model = model.merge_and_unload()

        return model, processor

    from transformers import AutoTokenizer

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        weights_path, add_eos_token=True, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token_id is None:
        if "<|padding|>" in tokenizer.get_vocab():
            # StabilityLM specific fix
            tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        elif tokenizer.unk_token is not None:
            print(
                "Tokenizer does not have a pad token, we will use the unk token as pad token."
            )
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            print(
                "Tokenizer does not have a pad token. We will use the eos token as pad token."
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        print(
            f"Model {weights_path} is a causal language model. We will load it as a CausalLM model."
        )
        model_kwargs = {
            "pretrained_model_name_or_path": weights_path,
            "device_map": "auto" if force_auto_device_map else None,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "quantization_config": bnb_config,
            **quant_args,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Ensure that the padding token is added to the left of the input sequence.
        tokenizer.padding_side = "left"
    else:
        raise ValueError(
            f"Model {weights_path} of type {config.model_type} is not supported. "
            f"{LLM_ONLY_MODEL_ERROR}"
        )

    if lora_weights_name_or_path:
        from peft import PeftModel

        print(f"Loading pretrained LORA weights from {lora_weights_name_or_path}")
        model = PeftModel.from_pretrained(model, lora_weights_name_or_path)

        if quantization is None:
            # If we are not using quantization, we merge the LoRA layers into the model for faster inference.
            # This is not possible if we are using 4/8 bit quantization.
            model = model.merge_and_unload()

    return model, tokenizer
