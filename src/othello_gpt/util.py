from typing import List
import huggingface_hub as hf
from pathlib import Path

import einops
import torch as t
from bidict import bidict
from jaxtyping import Int, Float
from torch import Tensor
from transformer_lens import HookedTransformerConfig, HookedTransformer

from othello_gpt.model.nanoGPT import GPTConfig, GPT

PAD_TOKEN = -1


def load_probes(probe_dir: Path, device, w_u=None, w_e=None, normed=True, combos=[]):
    probe_names = {
        "tem": "probe_tem_20250221_012810.pt",
        (None, "c"): "probe_cap_20250221_025002.pt",
        (None, "l"): "probe_legal_20250221_014256.pt",
        ("pt", "pe", "pm"): "probe_ptem_20250221_021621.pt",
        "d": "probe_dir_20250221_145729.pt",
    }

    probes = {}
    for names, file in probe_names.items():
        probe = t.load(
            probe_dir / file, weights_only=True, map_location=device
        ).detach()

        for i, n in enumerate(names):
            if n is None:
                continue
            probes[n] = probe[..., i, :]

    n_probe_layers = probes["t"].shape[-1]
    if w_u is not None:
        probes["u"] = einops.repeat(
            vocab_to_board(w_u),
            "d_model row col -> d_model row col n",
            n=n_probe_layers,
        ).flatten(1, 2)
    if w_e is not None:
        probes["b"] = einops.repeat(
            vocab_to_board(w_e),
            "d_model row col -> d_model row col n",
            n=n_probe_layers,
        ).flatten(1, 2)

    if normed:
        for k in probes:
            norm = probes[k].norm(dim=0, keepdim=True)
            norm = norm.nan_to_num(1)
            probes[k] = probes[k] / norm

    for i in range(len(combos)):
        combo = combos[i]
        if combo[0] not in "+-":
            combo = "+" + combo

        signs = []
        components = []
        for c in combo:
            if c in "+-":
                signs.append(c)
                components.append("")
            else:
                components[-1] += c

        probes[combo] = t.zeros_like(probes[components[0]])
        for s, k in zip(signs, components):
            if s == "+":
                probes[combo] += probes[k].clone()
            elif s == "-":
                probes[combo] -= probes[k].clone()

        if normed:
            norm = probes[combo].norm(dim=0, keepdim=True)
            norm = norm.nan_to_num(1)
            probes[combo] = probes[combo] / norm

    return probes


def load_model(device, name: str = "awonga/othello-gpt-30l", eval: bool = True):
    class HubGPT(GPT, hf.PyTorchModelHubMixin):
        pass

    if name == "awonga/othello-gpt":
        size = 6
        n_layer = 2
        n_head = 4
        n_embd = 256
        bias = True
    elif name == "awonga/othello-gpt-30l":
        size = 6
        n_layer = 30
        n_head = 2
        n_embd = 108
        bias = False
    elif name == "awonga/othello-gpt-7M":
        size = 6
        n_layer = 30
        n_head = 8
        n_embd = 144
        bias = True
    else:
        raise ValueError(name)

    nano_cfg = GPTConfig(
        block_size=(size * size - 4) - 1,
        vocab_size=size * size - 4,  # no pad
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=bias,
    )
    hooked_cfg = HookedTransformerConfig(
        n_layers=nano_cfg.n_layer,
        d_model=nano_cfg.n_embd,
        n_ctx=nano_cfg.block_size,
        d_head=nano_cfg.n_embd // nano_cfg.n_head,
        n_heads=nano_cfg.n_head,
        d_vocab=nano_cfg.vocab_size,
        act_fn="gelu",
        normalization_type="LN",
        device=device,
    )

    model = HubGPT.from_pretrained(name, config=nano_cfg).to(device)
    state_dict = convert_nanogpt_to_transformer_lens_weights(
        model.state_dict(), nano_cfg, hooked_cfg
    )

    model = HookedTransformer(hooked_cfg)
    model.load_and_process_state_dict(state_dict)

    if eval:
        model.eval()
    return model


def get_all_squares(size: int):
    nw_middle_id = (size // 2 - 1) * size + (size // 2 - 1)
    initial_squares = set(
        [nw_middle_id, nw_middle_id + 1, nw_middle_id + size, nw_middle_id + size + 1]
    )
    all_squares = [i for i in range(size * size) if i not in initial_squares]
    return all_squares


def vocab_to_board(vocab: Float[t.Tensor, "... n_ctx"], size: int = 0, fill_value=None):
    if size == 0:
        size = int((vocab.shape[-1] + 4) ** 0.5)
    board_shape = (*vocab.shape[:-1], size, size)

    if fill_value is None:
        board = t.empty(board_shape)
    else:
        board = t.full(board_shape, fill_value)
    board = board.to(vocab.device)

    all_squares = get_all_squares(size)
    board.flatten(-2)[..., all_squares] = vocab.clone()

    return board


def get_id_to_token_id_map(size: int, pad_token: int | None = None):
    all_squares = get_all_squares(size)
    if pad_token is not None:
        all_squares = [pad_token] + all_squares
    id_to_token_id_map = bidict(
        {square_id: token_id for token_id, square_id in enumerate(all_squares)}
    )
    return id_to_token_id_map


def tokenize(history, size, pad_token=PAD_TOKEN):
    if isinstance(history[0], list):
        # TODO vectorise/parallelise
        return {
            "input_ids": [tokenize(h, size, pad_token)["input_ids"] for h in history]
        }
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
