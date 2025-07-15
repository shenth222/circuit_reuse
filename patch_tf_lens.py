from transformer_lens.HookedTransformer import HookedTransformer

import logging
from loguru import logger
import os
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from jaxtyping import Float, Int
from packaging import version
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

T = TypeVar("T", bound="HookedTransformer")

@classmethod
def from_pretrained(
    cls: Type[T],
    model_name: str,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    refactor_factored_attn_matrices: bool = False,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    hf_model: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    move_to_device: bool = True,
    fold_value_biases: bool = True,
    default_prepend_bos: Optional[bool] = None,
    default_padding_side: Literal["left", "right"] = "right",
    dtype="float32",
    first_n_layers: Optional[int] = None,
    **from_pretrained_kwargs,
) -> T:
    if model_name.lower().startswith("t5"):
        raise RuntimeError(
            "Execution stopped: Please use HookedEncoderDecoder to load T5 models instead of HookedTransformer."
        )

    assert not (
        from_pretrained_kwargs.get("load_in_8bit", False)
        or from_pretrained_kwargs.get("load_in_4bit", False)
    ), "Quantization not supported"

    if hf_model is not None:
        assert hf_model.config is not None
        hf_cfg = hf_model.config.to_dict()
        qc = hf_cfg.get("quantization_config", {})
        load_in_4bit = qc.get("load_in_4bit", False)
        load_in_8bit = qc.get("load_in_8bit", False)
        quant_method = qc.get("quant_method", "")
        assert not load_in_8bit, "8-bit quantization is not supported"
        assert not (
            load_in_4bit and (version.parse(torch.__version__) < version.parse("2.1.1"))
        ), "Quantization is only supported for torch versions >= 2.1.1"
        assert not (
            load_in_4bit and ("llama" not in model_name.lower())
        ), "Quantization is only supported for Llama models"
        if load_in_4bit:
            assert (
                qc.get("quant_method", "") == "bitsandbytes"
            ), "Only bitsandbytes quantization is supported"
    else:
        hf_cfg = {}

    if isinstance(dtype, str):
        # Convert from string to a torch dtype
        dtype = DTYPE_FROM_STRING[dtype]
    if "torch_dtype" in from_pretrained_kwargs:
        # For backwards compatibility with the previous way to do low precision loading
        # This should maybe check the user did not explicitly set dtype *and* torch_dtype
        dtype = from_pretrained_kwargs["torch_dtype"]

    if (
        (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
        or dtype == torch.float16
    ) and device in ["cpu", None]:
        logger.warning("float16 models may not work on CPU. Consider using a GPU or bfloat16.")

    # Get the model name used in HuggingFace, rather than the alias.
    official_model_name = loading.get_official_model_name(model_name)

    # Load the config into an HookedTransformerConfig object. If loading from a
    # checkpoint, the config object will contain the information about the
    # checkpoint
    cfg = loading.get_pretrained_model_config(
        official_model_name,
        hf_cfg=hf_cfg,
        checkpoint_index=checkpoint_index,
        checkpoint_value=checkpoint_value,
        fold_ln=fold_ln,
        device=device,
        n_devices=n_devices,
        default_prepend_bos=default_prepend_bos,
        dtype=dtype,
        first_n_layers=first_n_layers,
        **from_pretrained_kwargs,
    )

    if cfg.positional_embedding_type == "shortformer":
        if fold_ln:
            logger.warning(
                "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_"
                "ln=False instead."
            )
            fold_ln = False
        if center_unembed:
            logger.warning(
                "You tried to specify center_unembed=True for a shortformer model, but this can't be done! "
                "Setting center_unembed=False instead."
            )
            center_unembed = False
        if center_writing_weights:
            logger.warning(
                "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! "
                "Setting center_writing_weights=False instead."
            )
            center_writing_weights = False
    if center_unembed and cfg.output_logits_soft_cap > 0.0:
        logger.warning(
            "You tried to specify center_unembed=True for a model using logit softcap, but this can't be done! Softcapping is not invariant upon adding a constant "
            "Setting center_unembed=False instead."
        )
        center_unembed = False

    # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to
    # match the HookedTransformer parameter names.
    state_dict = loading.get_pretrained_state_dict(
        official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
    )

    # Create the HookedTransformer object
    model = cls(
        cfg,
        tokenizer,
        move_to_device=False,
        default_padding_side=default_padding_side,
    )

    model.load_and_process_state_dict(
        state_dict,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
    )

    if move_to_device:
        model.move_model_modules_to_device()

    print(f"Loaded pretrained model {model_name} into HookedTransformer")

    return model

from transformer_lens.loading_from_pretrained import make_model_alias_map, get_official_model_name, OFFICIAL_MODEL_NAMES, convert_hf_model_config
import os
def get_official_model_name(model_name: str):
    """
    Returns the official model name for a given model name (or alias).
    """
    if os.path.exists(model_name) and os.path.isdir(model_name):
        return model_name
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower(), None)
    if official_model_name is None:
        raise ValueError(
            f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}"
        )
    return official_model_name

def convert_hf_model_config(model_name: str, **kwargs: Any):
    if os.path.exists(model_name) and os.path.isdir(model_name):
        logger.info("Loading model config from local directory")
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)
    
        # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            **kwargs,
        )
    elif "gemma-2" in official_model_name.lower():
        architecture = "Gemma2ForCausalLM"
    elif "gemma" in official_model_name.lower():
        architecture = "GemmaForCausalLM"
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            token=huggingface_token if len(huggingface_token) > 0 else None,
            **kwargs,
        )
        architecture = hf_config.architectures[0]

    cfg_dict: dict[str, Any]
    if official_model_name.startswith(
        ("llama-7b", "meta-llama/Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("codellama"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Llama-3.2-1B" in official_model_name:
        cfg_dict = {
            "d_model": 2048,
            "d_head": 64,
            "n_heads": 32,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 64,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.2-3B" in official_model_name:
        cfg_dict = {
            "d_model": 3072,
            "d_head": 128,
            "n_heads": 24,
            "d_mlp": 8192,
            "n_layers": 28,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.1-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.1-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "rotary_adjacent_pairs": True,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        # All supported Bert architectures have the same config,
        # so we can use the BertForMaskedLM config for all of them
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    elif architecture == "MistralForCausalLM":
        use_local_attn = True if hf_config.sliding_window else False
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": (
                hf_config.head_dim
                if hasattr(hf_config, "head_dim")
                and hf_config.head_dim is not None
                and hf_config.head_dim > 0
                else hf_config.hidden_size // hf_config.num_attention_heads
            ),
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "window_size": hf_config.sliding_window,  # None if no sliding window was used
            "attn_types": ["local"] * hf_config.num_hidden_layers if use_local_attn else None,
            "eps": hf_config.rms_norm_eps,
            "rotary_base": hf_config.rope_theta,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "use_local_attn": use_local_attn,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif architecture == "MixtralForCausalLM":
        cfg_dict = {
            "dtype": torch.bfloat16,
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
            "attn_types": ["global"] * 32,
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "gated_mlp": True,
            "use_local_attn": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "num_experts": hf_config.num_local_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
        }
    elif architecture == "BloomForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
            "default_prepend_bos": False,
        }
    elif architecture == "GPT2LMHeadCustomModel":
        # santacoder
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "trust_remote_code": "santacoder"
            in official_model_name,  # Only santacoder needs trust_remote_code
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "tokenizer_prepends_bos": True,
            "trust_remote_code": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
    elif architecture == "Qwen3ForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            and hf_config.head_dim is not None
            and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.head_dim
            if hasattr(hf_config, "head_dim") and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
            "use_qk_norm": True,
            "trust_remote_code": True,
        }
    elif architecture == "PhiForCausalLM":
        # Architecture for microsoft/phi models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
        partial_rotary_factor = hf_config.partial_rotary_factor
        cfg_dict["rotary_dim"] = round(partial_rotary_factor * cfg_dict["d_head"])
    elif architecture == "Phi3ForCausalLM":
        # Architecture for microsoft/phi3 models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "gated_mlp": True,
            "parallel_attn_mlp": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        }

    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-2-2b"):
        # Architecture for Gemma-2 2b and Gemma-2 2b Instruct models
        cfg_dict = {
            "d_model": 2304,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 9216,
            "n_layers": 26,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-9b"):
        # Architecture for Gemma-2 9b and Gemma-2 9b Instruct models
        cfg_dict = {
            "d_model": 3584,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 14336,
            "n_layers": 42,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-27b"):
        # Architecture for Gemma-2 27b and Gemma-2 27b Instruct models
        cfg_dict = {
            "d_model": 4608,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 36864,
            "n_layers": 46,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "attn_scale": 12.0,
            "n_key_value_heads": 16,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 23,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif architecture == "T5ForConditionalGeneration":
        cfg_dict = {
            "d_model": hf_config.d_model,
            "d_head": hf_config.d_kv,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.d_ff,
            "d_vocab": hf_config.vocab_size,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_length,
            "eps": hf_config.layer_norm_epsilon,
            "act_fn": hf_config.feed_forward_proj,
            "positional_embedding_type": "relative_positional_bias",
            "relative_attention_max_distance": hf_config.relative_attention_max_distance,
            "relative_attention_num_buckets": hf_config.relative_attention_num_buckets,
            "decoder_start_token_id": hf_config.decoder_start_token_id,
            "attention_dir": "bidirectional",
            "use_attn_scale": False,
            "tie_word_embeddings": hf_config.tie_word_embeddings,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict

def patch_transformer_lens():
    HookedTransformer.from_pretrained = from_pretrained
    loading.get_official_model_name = get_official_model_name
    loading.convert_hf_model_config = convert_hf_model_config

