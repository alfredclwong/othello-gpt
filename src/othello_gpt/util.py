from typing import List

import einops
import torch as t
from bidict import bidict
from jaxtyping import Int
from torch import Tensor
from transformer_lens import HookedTransformerConfig

from othello_gpt.model.nanoGPT import GPTConfig

PAD_TOKEN = -1


def get_all_squares(size: int):
    nw_middle_id = (size // 2 - 1) * size + (size // 2 - 1)
    initial_squares = set(
        [nw_middle_id, nw_middle_id + 1, nw_middle_id + size, nw_middle_id + size + 1]
    )
    all_squares = [i for i in range(size * size) if i not in initial_squares]
    return all_squares


def get_id_to_token_id_map(size: int, pad_token: int | None = None):
    all_squares = get_all_squares(size)
    if pad_token is not None:
        all_squares = [pad_token] + all_squares
    id_to_token_id_map = bidict(
        {
            square_id: token_id
            for token_id, square_id in enumerate(all_squares)
        }
    )
    return id_to_token_id_map


def tokenize(history, size, pad_token=PAD_TOKEN):
    if isinstance(history[0], list):
        # TODO vectorise/parallelise
        return {"input_ids": [tokenize(h, size, pad_token)["input_ids"] for h in history]}
    id_to_token_id_map = get_id_to_token_id_map(size, pad_token)
    return {"input_ids": [id_to_token_id_map[i] for i in history]}


def decode(token_ids, size, pad_token=PAD_TOKEN):
    id_to_token_id_map = get_id_to_token_id_map(size, pad_token)
    return {"square_ids": [id_to_token_id_map.inverse[i] for i in token_ids]}


def pad_batch(
    batch: List[List[int]], max_len: int, pad_token: int = PAD_TOKEN
) -> Int[Tensor, "batch max_len"]:
    padded_batch = t.full((len(batch), max_len), pad_token)
    for i, seq in enumerate(batch):
        padded_batch[i, -len(seq) :] = t.tensor(seq)
    return padded_batch


# https://github.com/adamkarvonen/chess_llm_interpretability/blob/0f61e667fb8a809deda29e5db6c113a0a88f9998/model_setup.py#L49
def convert_nanogpt_to_transformer_lens_weights(
    old_state_dict, nano_cfg: GPTConfig, hooked_cfg: HookedTransformerConfig
):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""
    bias = nano_cfg.bias

    # Nanogpt models saved after torch.compile() have this unwanted prefix
    # This is a simple way to remove it
    unwanted_prefix = "_orig_mod."
    for k, v in list(old_state_dict.items()):
        if k.startswith(unwanted_prefix):
            old_state_dict[k[len(unwanted_prefix) :]] = old_state_dict.pop(k)

    new_state_dict = {}
    new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = t.zeros_like(
        old_state_dict["transformer.ln_f.weight"]
    )
    new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T

    if bias:
        new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]

    for layer in range(hooked_cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[
            f"{layer_key}.ln_1.weight"
        ]
        # A bias of zeros is required for folding layer norm
        new_state_dict[f"blocks.{layer}.ln1.b"] = t.zeros_like(
            old_state_dict[f"{layer_key}.ln_1.weight"]
        )
        new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[
            f"{layer_key}.ln_2.weight"
        ]
        new_state_dict[f"blocks.{layer}.ln2.b"] = t.zeros_like(
            old_state_dict[f"{layer_key}.ln_2.weight"]
        )

        W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = t.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=hooked_cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=hooked_cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=hooked_cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=hooked_cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ].T
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ].T

        if bias:
            new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[
                f"{layer_key}.ln_1.bias"
            ]
            new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[
                f"{layer_key}.ln_2.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = t.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=hooked_cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=hooked_cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=hooked_cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]

    return new_state_dict
