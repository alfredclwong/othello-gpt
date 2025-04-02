# %%
from pathlib import Path

import torch as t
from datasets import Dataset, load_dataset
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jaxtyping import Float
from torch.types import Tensor

from othello_gpt.data.vis import move_id_to_text, plot_game
from othello_gpt.util import load_model, load_probes, get_all_squares
from othello_gpt.model.sae import OthelloSAE, OthelloSAEConfig
import datasets
from itertools import product
import numpy as np

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

model_version = "300k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)

dataset_dict = load_dataset("awonga/othello-gpt")
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

size = 6
all_squares = get_all_squares(size)
actually_all_squares = list(range(size * size))
corners = [0, size - 1, size * (size - 1), size * size - 1]
non_corners = [i for i in range(size * size) if i not in corners]
edges = [
    y * size + x
    for y in range(size)
    for x in range(size)
    if x * y == 0 or x == size - 1 or y == size - 1
]
non_edges = [i for i in range(size * size) if i not in edges]

# %%
padded_W_pos = t.full((size * size, model.W_pos.shape[1]), t.nan, device=device)
padded_W_pos[: model.W_pos.shape[0], :] = model.W_pos
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=padded_W_pos.T.detach(),
    combos=[
        "+t-m",
        "+pee-ee",
    ],
    model_version=model_version,
)
probes["pos"] = t.nn.functional.pad(
    probes["pos"], (0, 0, 0, size * size - probes["pos"].shape[1]), value=float("nan")
)

probe_keys = ["ee", "+t-m", "c", "mov", "b", "u", "p", "l", "pos"]
probes_normed: Float[Tensor, "d_model n_probe"] = t.cat(
    [probes[k][..., 3] for k in probe_keys], dim=1
)
probe_suffixes = {
    k: [
        f"{i if k in ['p', 'pr'] else move_id_to_text(i, size)}"
        for i in actually_all_squares
    ]
    for k in probe_keys
}
probe_bases_labels = [f"{k}_{s}" for k in probe_keys for s in probe_suffixes[k]]

{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
sae_cfg = OthelloSAEConfig(
    d_in=model.cfg.d_model,
    d_sae=1024,
    hook_layers=list(range(model.cfg.n_layers)),
    hook_suffixes=["attn.hook_z", "hook_mlp_out"],
)
sae = OthelloSAE.from_pretrained(
    f"{model_name}-sae",
    sae_cfg=sae_cfg,
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    device=device,
)
eval_dict = sae.evaluate()
dataset = sae.test_dataset
with t.inference_mode():
    test_forward_dict = sae.forward_dataset(dataset)

# %%
# Result visualisation
# 1. Loss recovered
# 2. x norms, error norms
# 3. Latent activity
# 4. Self colinearity, linear probe colinearity
# 5. Max activating datasets

# %%
# Analysis
# 1. Binary feature pairs
# 2. Feature geometry
# 3. Feature circuits
# 4.

# %%
{k: eval_dict[k] for k in sorted(eval_dict.keys())}

# %%
x_norms = dict(
    zip(sae.hook_names, test_forward_dict["x"].norm(dim=-1).mean(0).tolist())
)
x_recon_norms = dict(
    zip(sae.hook_names, test_forward_dict["x_recon"].norm(dim=-1).mean(0).tolist())
)
# L5 mlp out is massive! hypothesis: empty and legal are same direction, but legal gets
# evidence-boosted to a much higher magnitude for softmax
# so should we have magnitude-range based features? don't think this is necessary - we
# can have both empty and legal being co-active above threshold, this actually prevents splitting
x_norms, x_recon_norms

# %%
acts_post = test_forward_dict["acts_post"].reshape(
    -1, sae.model.cfg.n_ctx, sae.n_sae, sae.cfg.d_sae
)
frac_active = (acts_post > 1e-8).float().flatten(0, 1).mean(0)
sorted_latent_idxs = t.argsort(frac_active, dim=-1, descending=True)

fig = make_subplots(
    rows=sae.n_sae,
    cols=2,
    shared_xaxes=True,
    subplot_titles=[
        f"{h} {t}" for h, t in product(sae.hook_names, ["acts", "frac_active"])
    ],
)
for i, hook_name in enumerate(sae.hook_names):
    fig.add_trace(
        go.Heatmap(
            z=acts_post[:10, :, i, sorted_latent_idxs[i]].flatten(0, 1).cpu(),
            showscale=False,
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=frac_active[i, sorted_latent_idxs[i]].cpu(), showlegend=False),
        row=i + 1,
        col=2,
    )
fig.update_layout(
    height=200 * sae.n_sae,
)
fig.show()

# %%
latents_normed: Float[Tensor, "n_sae d_sae d_in"] = (
    sae.W_dec_normalized.detach().clone()
)
for i, hook_name in enumerate(sae.hook_names):
    latents_normed[i] = latents_normed[i, sorted_latent_idxs[i]]
    if "hook_z" in hook_name:
        hook_layer = int(hook_name[len("blocks.")][0])
        latents_normed[i] @= sae.model.W_O[hook_layer].flatten(0, 1)
latents_normed /= latents_normed.norm(dim=-1, keepdim=True)
flat_latents_normed = latents_normed.flatten(0, 1)

self_colinearity_matrix = flat_latents_normed @ flat_latents_normed.T
self_colinearity_matrix = t.tril(self_colinearity_matrix, diagonal=-1)
colinearity_matrix = latents_normed @ probes_normed
flat_colinearity_matrix = colinearity_matrix.flatten(0, 1)

# Generate random vectors for comparison
random_vectors = t.randn_like(flat_latents_normed)
random_vectors /= random_vectors.norm(dim=-1, keepdim=True)
random_colinearity_matrix = (random_vectors @ random_vectors.T).abs()
random_colinearity_matrix = t.tril(random_colinearity_matrix, diagonal=-1)

# Calculate the maximum values along the 0th dimension
self_max_values = self_colinearity_matrix.max(0)[0].cpu().numpy()
max_values = flat_colinearity_matrix.max(0)[0].cpu().numpy()
random_max_values = random_colinearity_matrix.max(0)[0].cpu().numpy()

# %%
# # Create a DataFrame for the histogram
# self_df = pd.DataFrame({"Maximum Colinearity": self_max_values})
# df = pd.DataFrame({"Maximum Colinearity": max_values})
# random_df = pd.DataFrame({"Maximum Colinearity": random_max_values})

# random_fig = px.histogram(
#     random_df,
#     x="Maximum Colinearity",
#     nbins=50,
#     title="Histogram of Maximum Random Colinearity Values",
#     labels={"Maximum Colinearity": "Maximum Colinearity"},
# )
# random_fig.update_layout(
#     xaxis_title="Maximum Colinearity",
#     yaxis_title="Frequency",
#     bargap=0.1,
# )
# random_fig.show()

# fig = px.histogram(
#     df,
#     x="Maximum Colinearity",
#     nbins=50,
#     title="Histogram of Maximum Colinearity Values",
#     labels={"Maximum Colinearity": "Maximum Colinearity"},
# )
# fig.update_layout(
#     xaxis_title="Maximum Colinearity",
#     yaxis_title="Frequency",
#     bargap=0.1,
# )
# fig.show()

# fig = px.histogram(
#     self_df,
#     x="Maximum Colinearity",
#     nbins=50,
#     title="Histogram of Maximum Self-Colinearity Values",
#     labels={"Maximum Colinearity": "Maximum Colinearity"},
# )
# fig.update_layout(
#     xaxis_title="Maximum Colinearity",
#     yaxis_title="Frequency",
#     bargap=0.1,
# )
# fig.show()

# fig = go.Figure()
# fig.add_trace(
#     go.Heatmap(
#         z=flat_colinearity_matrix.cpu(),
#         x=probe_bases_labels,
#         # y=sorted_latent_idxs.tolist(),
#     )
# )
# fig.update_layout(title="Latent alignment with linear probes")
# fig.show()

# %%
top_k = 3
abs_colinearity = colinearity_matrix.abs().nan_to_num(0)
for i, hook_name in enumerate(sae.hook_names):
    _, top_indices = t.topk(abs_colinearity[i].flatten(), k=top_k)
    for k, idx in enumerate(top_indices):
        colinearity = colinearity_matrix[i].flatten()[idx].item()
        latent_idx, probe_idx = divmod(idx.item(), colinearity_matrix.shape[-1])
        pct_active = frac_active[i, sorted_latent_idxs[i, latent_idx]]
        print(
            f"{hook_name} #{k}: {colinearity=:.2f}, {latent_idx=}, {probe_bases_labels[probe_idx]}, {pct_active=:.2%}"
        )

# %%
# Given (sae_idx, latent_idx), find the max activating datasets and plot the games with activation values
top_k = 3
sae_idx = 1
latent_idx = sorted_latent_idxs[sae_idx, 326]
max_act_per_game = acts_post[..., sae_idx, latent_idx].max(dim=1)[0]
_, top_game_idxs = t.topk(max_act_per_game, k=top_k)
flat_dataset = datasets.concatenate_datasets([Dataset.from_dict(d) for d in dataset])
for game_idx in top_game_idxs.tolist():
    game_acts = acts_post[game_idx, :, sae_idx, latent_idx]
    plot_game(
        flat_dataset[game_idx],
        subplot_titles=[
            f"<b style='color:red;'>{act:.2f}</b>" if act > 0 else ""
            for act in game_acts.tolist()
        ],
    )

# %%
latent_normed = latents_normed[sae_idx, 326]
latent_probe_similarities = latent_normed @ probes_normed
latent_probe_similarities = latent_probe_similarities.reshape(len(probe_keys), -1).T
latent_probe_df = pd.DataFrame(
    latent_probe_similarities.cpu().numpy(),
    columns=probe_keys,
    index=list(enumerate(move_id_to_text(i, size) for i in actually_all_squares)),
)


# Highlight the top 3 absolute values in the DataFrame
def highlight_top3(s):
    is_top3 = s.abs().nlargest(3).index
    return ["background-color: yellow" if i in is_top3 else "" for i in s.index]


latent_probe_df = latent_probe_df.style.apply(highlight_top3, axis=0)
latent_probe_df

# %%
sae_layer, is_mlp = divmod(3, 2)

# %%
# sae_layer, is_mlp = divmod(sae_idx, 2)
for sae_layer, is_mlp in product(range(sae.model.cfg.n_layers), [0, 1]):
    upstream_labels = [
        *[
            f"Q{l}H{h}D{d}"
            for l in range(sae_layer + 1, sae.model.cfg.n_layers)
            for h in range(sae.model.cfg.n_heads)
            for d in range(sae.model.cfg.d_head)
        ],
        *[
            f"K{l}H{h}D{d}"
            for l in range(sae_layer + 1, sae.model.cfg.n_layers)
            for h in range(sae.model.cfg.n_heads)
            for d in range(sae.model.cfg.d_head)
        ],
        *[
            f"V{l}H{h}D{d}"
            for l in range(sae_layer + 1, sae.model.cfg.n_layers)
            for h in range(sae.model.cfg.n_heads)
            for d in range(sae.model.cfg.d_head)
        ],
        *[
            f"M{l}N{n}"
            for l in range(sae_layer + is_mlp, sae.model.cfg.n_layers)
            for n in range(sae.model.cfg.d_mlp)
        ],
        *[
            f"U_{move_id_to_text(i, size)}"
            for i in all_squares
        ]
    ]
    upstream_weights = [
        model.W_Q[sae_layer + 1 :],
        model.W_K[sae_layer + 1 :],
        model.W_V[sae_layer + 1 :],
        model.W_in[sae_layer + is_mlp :],
        model.W_U.unsqueeze(0),
    ]
    upstream_weights = t.cat(
        [w.transpose(-2, -1).flatten(0, -2) for w in upstream_weights], dim=0
    )
    upstream_weights /= upstream_weights.norm(dim=-1, keepdim=True)

    upstream_activations = latents_normed[sae_idx] @ upstream_weights.T
    upstream_activations = upstream_activations.where(
        upstream_activations.abs() > 0.5, t.nan
    )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=upstream_activations.cpu(),
            x=upstream_labels,
        )
    )
    fig.show()

# %%
r = t.randn_like(latents_normed[sae_idx])
r /= r.norm(dim=-1, keepdim=True)
random_upstream_activations = r @ upstream_weights.T
random_upstream_activations = random_upstream_activations.where(
    random_upstream_activations.abs() > 0.5, t.nan
)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=random_upstream_activations.cpu(),
        x=upstream_labels,
    )
)
fig.show()
