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
from transformer_lens import HookedTransformer
from typing import List, Tuple

from othello_gpt.data.vis import plot_in_basis, move_id_to_text
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

# hf.login((root_dir / "secret.txt").read_text())
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
# version = "6M"
version = "4M"
model = load_model(device, f"awonga/othello-gpt-{version}")

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    # combos=["+t-m", "t+m", "+m-pt", "+t-pt", "+t-pm", "+m-pm"],
    combos=[
        # "+t-m",
        "+pee-ee",
    ],
    model_version=version,
)
# probes["r"] = t.randn_like(probes["u"])
# probes["r"] /= probes["r"].norm(dim=0, keepdim=True)
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
model.b_K

# %%
layers = list(range(n_layer))
layers = [1]
heads = list(range(n_head))
heads = [0]
labels = [
    f"L{l}H{h}D{d}"
    for l in layers
    for h in heads
    for d in range(d_head)
]
probe_key = "tm"
# probe_key = "+pee-ee"
w_k_ee = einops.einsum(
    model.W_K[layers][:, heads].transpose(-2, -1).flatten(0, -2),
    probes[probe_key][..., 4],
    "lhd d_model, d_model n_out -> lhd n_out"
)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=w_k_ee.detach().cpu(),
        x=[move_id_to_text(i, size) for i in range(size * size)],
        y=labels,
    )
)
fig.update_layout(
    title=f"W_K in {probe_key.upper()} space",
    height=1200,
)
fig.show()

# %%
layers = [2]
heads = [1]
labels = [
    f"L{l}H{h}D{d}"
    for l in layers
    for h in heads
    for d in range(d_head)
]
# probe_key = "b"
probe_key = "+pee-ee"
w_q_ee = einops.einsum(
    model.W_Q[layers, heads].transpose(-2, -1).flatten(0, -2),
    probes[probe_key][..., 4],
    "lhd d_model, d_model n_out -> lhd n_out"
)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=w_q_ee.detach().cpu(),
        x=[move_id_to_text(i, size) for i in range(size * size)],
        y=labels,
    )
)
fig.update_layout(
    title=f"W_Q in {probe_key.upper()} space",
    height=1200,
)
fig.show()

# %%
labels = [f"L{l}N{n}" for l in range(n_layer) for n in range(n_neuron)]
w_in_l2_ee = einops.einsum(
    w_in, probes["ee"][..., 4], "layer neuron d_model, d_model n_out -> layer neuron n_out"
).flatten(0, 1)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=w_in_l2_ee.detach().cpu(),
        x=[move_id_to_text(i, size) for i in range(size * size)],
        y=labels,
    )
)
fig.update_layout(
    height=800,
    title="W_in in EE space"
)
fig.show()

# %%
labels = [f"M{l} N{n}" for l in [2] for n in range(n_neuron)]
probe_layer = 5
# probe_name = "t-m"
# probe_name = "e"
# probe_name = "+m-pe"
# probe_name = "l"
probe_name = "ee"
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
# TODO make fn for plotting all these together
# TODO graph discovery
# Show a modular neuron circuit for legality
neuron = (-1, 140)
probe_layer = (neuron[0] + 1) * 2
neuron_probes = {
    "w_in": list("tem"),
    "w_out": list("temlu"),
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
neuron = (25, 121)
probe_layer = (neuron[0] + 1) * 2
neuron_probes = {
    "w_in": ["pt", "pe", "pm", "t", "e", "m", "c"],
    "w_out": ["+m-pt", "+t-pt"],
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
# Show a modular neuron circuit for PT -> M
neuron = (26, 360)
probe_layer = (neuron[0] + 1) * 2
neuron_probes = {
    "w_in": ["pt"],
    "w_out": ["m"],
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
# Show a modular neuron circuit for PE -> E
neuron = (22, 107)
probe_layer = (neuron[0] + 1) * 2
neuron_probes = {
    "w_in": probes.keys(),
    "w_out": probes.keys(),
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
probes["t"].shape


# %%
def probe_neurons(
    model: HookedTransformer,
    neurons: List[Tuple[int, int]],
    in_probes: Float[t.Tensor, "probe d_model row col"],
    out_probes: Float[t.Tensor, "probe d_model row col"],
):
    # plot several neurons' w_in and w_out weights transformed into various probe directions
    # TODO probe directions might not map to a board!
    n_neurons = len(neurons)
    ls, ns = zip(*neurons)

    w_in: Float[t.Tensor, "neuron d_model"] = model.W_in[ls, :, ns]
    w_out: Float[t.Tensor, "neuron d_model"] = model.W_out[ls, ns, :]

    w_in /= w_in.norm(-1, keepdim=True)
    w_out /= w_out.norm(-1, keepdim=True)

    fig = make_subplots(rows=2, cols=n_neurons)

    in_probes = t.stack()


probe_neurons(model, [(22, 107), (0, 0)], [], [])


# %%
# Form (hopefully disjoint) sets of modular circuits


# %%
def plot_neuron_excess_kurtosis(
    w_in: Float[t.Tensor, "n_layer n_neuron d_model"],
    w_out: Float[t.Tensor, "n_layer n_neuron d_model"],
    probe: Float[t.Tensor, "d_model n_out"],
    fig: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
):
    w_in_probed = einops.einsum(
        w_in,
        probe,
        "n_layer n_neuron d_model, d_model n_out -> n_layer n_neuron n_out",
    ).cpu()
    w_out_probed = einops.einsum(
        w_out,
        probe,
        "n_layer n_neuron d_model, d_model n_out -> n_layer n_neuron n_out",
    ).cpu()
    w_in_ekurt = kurtosis(w_in_probed, axis=2, fisher=False) - 3
    w_out_ekurt = kurtosis(w_out_probed, axis=2, fisher=False) - 3

    w_in_probed_flat = w_in_probed
    w_out_probed_flat = w_out_probed

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
fig.update_layout(height=1600, width=200 * len(probes))
fig.show()


# %%
# Get stats on where each neuron activates vs probes

# %%
# Conditional on A1 being strongly (un)predicted, which L1 neurons activated strongly?
# Which post-L0 residual stream directions activated the L1 neurons?
# How did the L0 block create these directions?

# %%
