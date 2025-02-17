# %%
import torch as t
from datasets import load_dataset
import huggingface_hub as hf
from pathlib import Path
import einops
import numpy as np
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.graph_objects as go
from typing import Union, List, Optional, Callable
from jaxtyping import Float
from transformer_lens import ActivationCache
import circuitsvis as cv
from IPython.display import HTML
import plotly.express as px
from plotly.subplots import make_subplots

from othello_gpt.data.vis import plot_game
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from scipy.stats import kurtosis
from othello_gpt.util import (
    convert_nanogpt_to_transformer_lens_weights,
    get_all_squares,
    pad_batch,
    get_id_to_token_id_map,
)
from othello_gpt.data.vis import move_id_to_text

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

hf.login((root_dir / "secret.txt").read_text())
dataset_dict = load_dataset("awonga/othello-gpt")

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

size = 6
all_squares = get_all_squares(size)

# %%
probe_names = {
    ("ot", "oe", "om"): "linear_probe_20250213_221416_mid_otem.pt",
    "tem": "linear_probe_20250212_141120_mid_tem.pt",
    "f": "linear_probe_20250213_000355_mid_flips.pt",
}
probes = {}
for names, file in probe_names.items():
    probe = t.load(probe_dir / file, weights_only=True, map_location=device).detach()
    probe /= probe.norm(dim=0, keepdim=True)
    for i, n in enumerate(names):
        probes[n] = probe[..., i, :]
{k: p.shape for k, p in probes.items()}

# %%
class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass


nano_cfg = GPTConfig(
    # block_size=(size * size - 4) * 2 - 1,
    block_size=(size * size - 4) - 1,
    # vocab_size=size * size - 4 + 2,  # pass and pad
    vocab_size=size * size - 4 + 1,  # pad
    n_layer=2,
    n_head=2,
    n_embd=128,
    dropout=0.0,
    bias=True,
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

model = HubGPT.from_pretrained("awonga/othello-gpt", config=nano_cfg).to(device)
state_dict = convert_nanogpt_to_transformer_lens_weights(
    model.state_dict(), nano_cfg, hooked_cfg
)
model = HookedTransformer(hooked_cfg)
model.load_and_process_state_dict(state_dict)
model.eval()
model.to(device)

# %%
def plot_in_basis(
    vectors: Float[t.Tensor, "n d_model"],
    probe: Float[t.Tensor, "d_model row col"],
    labels: List[str],
    filter_by: str = "",
    sort_by: str = "kurtosis",
    top_n: int = 0,
    title: str = "",
):
    # transform data vectors, e.g. neuron w_outs to a different basis, e.g. probe and visualise
    transformed_vectors = einops.einsum(
        vectors, probe,
        "n d_model, d_model row col -> n row col",
    )

    abs_max_indices = transformed_vectors.flatten(1).abs().max(dim=1)[1]
    abs_max_sign = t.sign(transformed_vectors.flatten(1)[t.arange(transformed_vectors.shape[0]), abs_max_indices])
    if filter_by == "pos":
        transformed_vectors = transformed_vectors[abs_max_sign > 0]
        labels = [labels[i] for i in range(len(labels)) if abs_max_sign[i] > 0]
    elif filter_by == "neg":
        transformed_vectors = transformed_vectors[abs_max_sign < 0]
        labels = [labels[i] for i in range(len(labels)) if abs_max_sign[i] < 0]

    if sort_by == "kurtosis":
        kurts = kurtosis(transformed_vectors, axis=(1, 2), fisher=False)
        sorted_indices = np.argsort(-kurts)
    else:
        sorted_indices = np.arange(transformed_vectors.shape[0])
    if top_n != 0:
        sorted_indices = sorted_indices[:top_n]
    transformed_vectors = transformed_vectors[sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    plot_game(
        {"boards": transformed_vectors},
        n_cols=5,
        hovertext=transformed_vectors,
        reversed=False,
        subplot_titles=labels,
        title=title,
    )


# %%
n_layer = model.cfg.n_layers
n_neuron = model.cfg.d_model * 4
w_out = model.W_out[:, :n_neuron].detach()
w_out /= w_out.norm(dim=-1, keepdim=True)
w_in = model.W_in[:, :n_neuron].transpose(1, 2).detach()
w_in /= w_in.norm(dim=-1, keepdim=True)

w_u_board = t.zeros((model.W_U.shape[0], size, size), device=device)
w_u_board.flatten(1)[:, all_squares] = model.W_U[:, 1:]
w_u_board /= w_u_board.norm(dim=0, keepdim=True)
w_u_board = w_u_board.nan_to_num().detach()
probes["u"] = einops.repeat(w_u_board, "d_model row col -> d_model row col n", n=n_layer*2+1)

# %%
labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]
plot_in_basis(w_in.flatten(0, 1), probes["u"][..., 2], labels, top_n=10, title="w_in . w_u (+ve)", filter_by="pos")
plot_in_basis(w_in.flatten(0, 1), probes["u"][..., 2], labels, top_n=10, title="w_in . w_u (-ve)", filter_by="neg")
plot_in_basis(w_out.flatten(0, 1), probes["u"][..., 2], labels, top_n=10, title="w_out . w_u (+ve)", filter_by="pos")
plot_in_basis(w_out.flatten(0, 1), probes["u"][..., 2], labels, top_n=10, title="w_out . w_u (-ve)", filter_by="neg")
plot_in_basis(w_in.flatten(0, 1), probes["f"][..., 2], labels, top_n=10, title="w_in . f2 (+ve)", filter_by="pos")
plot_in_basis(w_in.flatten(0, 1), probes["f"][..., 2], labels, top_n=10, title="w_in . f2 (-ve)", filter_by="neg")
plot_in_basis(w_out.flatten(0, 1), probes["f"][..., 2], labels, top_n=10, title="w_out . f2 (+ve)", filter_by="pos")
plot_in_basis(w_out.flatten(0, 1), probes["f"][..., 2], labels, top_n=10, title="w_out . f2 (-ve)", filter_by="neg")

# %%
t.randn((1, 2, 3, 4)).flatten(2).max(2)[0]

# %%
def plot_neuron_excess_kurtosis(
    w_in: Float[t.Tensor, "n_layer n_neuron d_model"],
    w_out: Float[t.Tensor, "n_layer n_neuron d_model"],
    probe: Float[t.Tensor, "d_model row col"],
    fig: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
):
    w_in_probed = einops.einsum(
        w_in, probe,
        "n_layer n_neuron d_model, d_model row col -> n_layer n_neuron row col"
    )
    w_out_probed = einops.einsum(
        w_out, probe,
        "n_layer n_neuron d_model, d_model row col -> n_layer n_neuron row col"
    )
    w_in_ekurt = kurtosis(w_in_probed, axis=(2, 3), fisher=False) - 3
    w_out_ekurt = kurtosis(w_out_probed, axis=(2, 3), fisher=False) - 3

    w_in_probed_flat = w_in_probed.flatten(2)
    w_out_probed_flat = w_out_probed.flatten(2)

    w_in_sign = t.sign(w_in_probed_flat[t.arange(w_in_probed_flat.shape[0])[:, None], t.arange(w_in_probed_flat.shape[1]), w_in_probed_flat.abs().argmax(dim=2)])
    w_out_sign = t.sign(w_out_probed_flat[t.arange(w_out_probed_flat.shape[0])[:, None], t.arange(w_out_probed_flat.shape[1]), w_out_probed_flat.abs().argmax(dim=2)])
    w_in_sign = w_in_sign.numpy()
    w_out_sign = w_out_sign.numpy()

    w_in_ekurt_pos = w_in_ekurt * (w_in_sign > 0)
    w_in_ekurt_neg = w_in_ekurt * (w_in_sign < 0)
    w_out_ekurt_pos = w_out_ekurt * (w_out_sign > 0)
    w_out_ekurt_neg = w_out_ekurt * (w_out_sign < 0)
    print(row, col, (w_in_sign < 0).sum(), (w_in_sign > 0).sum(), (w_out_sign < 0).sum(), (w_out_sign > 0).sum())

    if fig is None:
        fig = go.Figure()

    for layer in range(w_in_ekurt.shape[0]):
        fig.add_trace(
            go.Box(
                y=w_in_ekurt_neg[layer],
                name=f'L{layer} w_in-',
                boxmean=True,
            ), row=row, col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_in_ekurt_pos[layer],
                name=f'L{layer} w_in+',
                boxmean=True,
            ), row=row, col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_out_ekurt_neg[layer],
                name=f'L{layer} w_out-',
                boxmean=True,
            ), row=row, col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_out_ekurt_pos[layer],
                name=f'L{layer} w_out+',
                boxmean=True,
            ), row=row, col=col,
        )

    fig.update_layout(
        showlegend=False,
    )

# %%
layers = [2, 4]
fig = make_subplots(
    rows=len(layers),
    cols=len(probes),
    subplot_titles=[f"'{n}' L{layer}" for layer in layers for n in probes],
    shared_yaxes=True
)
for row, layer in enumerate(layers):
    for col, (n, p) in enumerate(probes.items()):
        plot_neuron_excess_kurtosis(w_in, w_out, p[..., layer], fig=fig, row=row+1, col=col+1)
fig.update_layout(height=800)
fig.show()

# %%
# Conditional on A1 being strongly (un)predicted, which L1 neurons activated strongly?
# Which post-L0 residual stream directions activated the L1 neurons?
# How did the L0 block create these directions?

# %%
model.W_Q.shape, model.W_K.shape, model.W_V.shape, model.W_O.shape  # n_layer n_head d_model d_head

# %%
def visualize_attention_patterns(
    heads: Union[List[int], int, Float[t.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: t.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[t.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = [move_id_to_text(t, size) for t in local_tokens]

    # Combine the patterns into a single tensor
    patterns: Float[t.Tensor, "head_index dest_pos src_pos"] = t.stack(
        patterns, dim=0
    )
    patterns *= (t.arange(patterns.shape[1]) + 1).unsqueeze(0).unsqueeze(-1)

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = cv.circuitsvis.attention.attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"

# %%
test_dataset = dataset_dict["test"].take(1000)
test_game = test_dataset[0]
test_input_ids = pad_batch([test_game["input_ids"]], max_len=model.cfg.n_ctx + 1)
test_logits, test_cache = model.run_with_cache(test_input_ids[:, :-1])
vis = visualize_attention_patterns(
    list(range(4)),
    test_cache,
    test_game["moves"],
)
HTML(vis)

# %%
w_o = model.W_O
w_v = model.W_V
w_ov = einops.einsum(
    w_o, w_v,
    "n_layer n_head d_head d_model_0, n_layer n_head d_model_1 d_head -> n_layer n_head d_model_0 d_model_1"
)
eigenvalues, _ = t.linalg.eig(w_ov.flatten(0, 1).detach().cpu())
eigenvalues /= eigenvalues.abs()


import plotly.express as px

# Prepare data for scatter plot
i = 3
eigenvalues_real = eigenvalues[i].real.numpy()
eigenvalues_imag = eigenvalues[i].imag.numpy()

# Create scatter plot using Plotly
fig = px.scatter(
    x=eigenvalues_real,
    y=eigenvalues_imag,
    labels={"x": "Real part", "y": "Imaginary part"},
    title="Scatter plot of eigenvalues"
)
fig.update_layout(xaxis_title="Real part", yaxis_title="Imaginary part")
fig.show()

# %%
