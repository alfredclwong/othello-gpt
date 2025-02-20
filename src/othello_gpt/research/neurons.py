# %%
import torch as t
from datasets import load_dataset
import huggingface_hub as hf
from pathlib import Path
import einops
import plotly.graph_objects as go
from typing import Optional
from jaxtyping import Float
from plotly.subplots import make_subplots

from othello_gpt.data.vis import plot_in_basis
from scipy.stats import kurtosis
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
)

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
model = load_model(device, "awonga/othello-gpt-7M")

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    combos=["t+m", "t-m"],
)
probes["r"] = t.randn_like(probes["u"])
probes["r"] /= probes["r"].norm(dim=0, keepdim=True)

{k: p.shape for k, p in probes.items()}

# %%
n_layer = model.cfg.n_layers
n_head = model.cfg.n_heads
d_head = model.cfg.d_head
n_neuron = model.cfg.d_model * 4

w_out = model.W_out[:, :n_neuron].detach()
w_out /= w_out.norm(dim=-1, keepdim=True)
w_in = model.W_in[:, :n_neuron].transpose(1, 2).detach()
w_in /= w_in.norm(dim=-1, keepdim=True)
neurons = {
    "w_in": w_in,
    "w_out": w_out,
}


# %%
labels = [f"M{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]
probe_layer = 50
# probe_name = "t-m"
probe_name = "l"
probe = probes[probe_name][..., probe_layer]
for s in ["pos", "neg"]:
    for n, w in neurons.items():
        plot_in_basis(
            w.flatten(0, 1),
            probe,
            labels,
            top_n=20,
            title=f"{n} . {probe_name}_{probe_layer} ({s})",
            filter_by=s,
            n_cols=10,
        )

# %%
# Show a modular neuron circuit in L1 for legality
neuron = (29, 508)
probe_layer = (neuron[0] + 1) * 2
neuron_probes = {
    "w_in": list("tem"),
    "w_out": list("teml"),
}
for n in neurons:
    for p in neuron_probes[n]:
        plot_in_basis(
            neurons[n][*neuron].unsqueeze(0),
            probes[p][..., probe_layer],
            labels=[f"M{neuron[0]} N{neuron[1]} {n} . {p}{probe_layer}"],
            n_cols=1,
        )

# %%
# Show a modular neuron circuit in L0 for captures

# %%
# Form (hopefully disjoint) sets of modular circuits

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
        w_in,
        probe,
        "n_layer n_neuron d_model, d_model row col -> n_layer n_neuron row col",
    ).cpu()
    w_out_probed = einops.einsum(
        w_out,
        probe,
        "n_layer n_neuron d_model, d_model row col -> n_layer n_neuron row col",
    ).cpu()
    w_in_ekurt = kurtosis(w_in_probed, axis=(2, 3), fisher=False) - 3
    w_out_ekurt = kurtosis(w_out_probed, axis=(2, 3), fisher=False) - 3

    w_in_probed_flat = w_in_probed.flatten(2)
    w_out_probed_flat = w_out_probed.flatten(2)

    w_in_sign = t.sign(
        w_in_probed_flat[
            t.arange(w_in_probed_flat.shape[0])[:, None],
            t.arange(w_in_probed_flat.shape[1]),
            w_in_probed_flat.abs().argmax(dim=2),
        ]
    )
    w_out_sign = t.sign(
        w_out_probed_flat[
            t.arange(w_out_probed_flat.shape[0])[:, None],
            t.arange(w_out_probed_flat.shape[1]),
            w_out_probed_flat.abs().argmax(dim=2),
        ]
    )
    w_in_sign = w_in_sign.numpy()
    w_out_sign = w_out_sign.numpy()

    w_in_ekurt_pos = w_in_ekurt * (w_in_sign > 0)
    w_in_ekurt_neg = w_in_ekurt * (w_in_sign < 0)
    w_out_ekurt_pos = w_out_ekurt * (w_out_sign > 0)
    w_out_ekurt_neg = w_out_ekurt * (w_out_sign < 0)
    print(
        row,
        col,
        (w_in_sign < 0).sum(),
        (w_in_sign > 0).sum(),
        (w_out_sign < 0).sum(),
        (w_out_sign > 0).sum(),
    )

    if fig is None:
        fig = go.Figure()

    for layer in range(w_in_ekurt.shape[0]):
        fig.add_trace(
            go.Box(
                y=w_in_ekurt_neg[layer],
                name=f"M{layer} w_in-",
                boxmean=True,
                boxpoints=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_in_ekurt_pos[layer],
                name=f"M{layer} w_in+",
                boxmean=True,
                boxpoints=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_out_ekurt_neg[layer],
                name=f"M{layer} w_out-",
                boxmean=True,
                boxpoints=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Box(
                y=w_out_ekurt_pos[layer],
                name=f"M{layer} w_out+",
                boxmean=True,
                boxpoints=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        showlegend=False,
    )


# %%
# Get stats on where each neuron activates vs probes


# %%
# layers = [2, 4]
probe_layers = [6, 30, 54]
fig = make_subplots(
    rows=len(probe_layers),
    cols=len(probes),
    subplot_titles=[f"'{n}' L{layer}" for layer in probe_layers for n in probes],
    shared_yaxes=True,
)
for row, layer in enumerate(probe_layers):
    for col, (n, p) in enumerate(probes.items()):
        plot_neuron_excess_kurtosis(
            w_in, w_out, p[..., layer], fig=fig, row=row + 1, col=col + 1
        )
fig.update_layout(height=1600)
fig.show()

# %%
# Conditional on A1 being strongly (un)predicted, which L1 neurons activated strongly?
# Which post-L0 residual stream directions activated the L1 neurons?
# How did the L0 block create these directions?
