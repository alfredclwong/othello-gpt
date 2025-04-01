# %%
from pathlib import Path

import torch as t
from datasets import Dataset, load_dataset
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from othello_gpt.data.vis import move_id_to_text, plot_game
from othello_gpt.util import load_model, load_probes, get_all_squares
from othello_gpt.model.sae import OthelloSAE, OthelloSAEConfig
import datasets
from itertools import product

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
with t.inference_mode():
    test_forward_dict = sae.forward_dataset(sae.train_dataset.take(12))

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
latents_normed = sae.W_dec_normalized.clone()
for i, hook_name in enumerate(sae.hook_names):
    latents_normed[i] = latents_normed[i, sorted_latent_idxs[i]]
    if "hook_z" in hook_name:
        hook_layer = int(hook_name[len("blocks.")][0])
        latents_normed[i] @= sae.model.W_O[hook_layer].flatten(0, 1)
latents_normed /= latents_normed.norm(dim=-1, keepdim=True)
latents_normed = latents_normed.flatten(0, 1)

colinear_keys = ["ee", "+t-m", "c", "mov", "b", "u", "p", "l", "pos"]
probes_normed = t.cat([probes[k][..., 3] for k in colinear_keys], dim=1)
probe_suffixes = {
    k: [
        f"{i if k in ['p', 'pr'] else move_id_to_text(i, size)}"
        for i in actually_all_squares
    ]
    for k in colinear_keys
}
probe_bases_labels = [f"{k}_{s}" for k in colinear_keys for s in probe_suffixes[k]]

self_colinearity_matrix = (latents_normed @ latents_normed.T).detach().cpu()
self_colinearity_matrix = t.tril(self_colinearity_matrix, diagonal=-1)
colinearity_matrix = (latents_normed @ probes_normed).detach().cpu()

# Generate random vectors for comparison
num_random_vectors = 1000
random_vectors = t.randn(num_random_vectors, latents_normed.shape[1])
random_vectors /= random_vectors.norm(dim=-1, keepdim=True)
random_colinearity_matrix = (random_vectors @ random_vectors.T).abs()
random_colinearity_matrix = t.tril(random_colinearity_matrix, diagonal=-1)

# Calculate the maximum values along the 0th dimension
self_max_values = self_colinearity_matrix.max(0)[0].cpu().numpy()
max_values = colinearity_matrix.max(0)[0].cpu().numpy()
random_max_values = random_colinearity_matrix.max(0)[0]

# Create a DataFrame for the histogram
self_df = pd.DataFrame({"Maximum Colinearity": self_max_values})
df = pd.DataFrame({"Maximum Colinearity": max_values})
random_df = pd.DataFrame({"Maximum Colinearity": random_max_values})

random_fig = px.histogram(
    random_df,
    x="Maximum Colinearity",
    nbins=50,
    title="Histogram of Maximum Random Colinearity Values",
    labels={"Maximum Colinearity": "Maximum Colinearity"},
)
random_fig.update_layout(
    xaxis_title="Maximum Colinearity",
    yaxis_title="Frequency",
    bargap=0.1,
)
random_fig.show()

fig = px.histogram(
    df,
    x="Maximum Colinearity",
    nbins=50,
    title="Histogram of Maximum Colinearity Values",
    labels={"Maximum Colinearity": "Maximum Colinearity"},
)
fig.update_layout(
    xaxis_title="Maximum Colinearity",
    yaxis_title="Frequency",
    bargap=0.1,
)
fig.show()

fig = px.histogram(
    self_df,
    x="Maximum Colinearity",
    nbins=50,
    title="Histogram of Maximum Self-Colinearity Values",
    labels={"Maximum Colinearity": "Maximum Colinearity"},
)
fig.update_layout(
    xaxis_title="Maximum Colinearity",
    yaxis_title="Frequency",
    bargap=0.1,
)
fig.show()

# # Find all pairs of latents with colinearity > 0.99
# threshold = 0.99
# high_colinearity_pairs = t.nonzero(self_colinearity_matrix > threshold, as_tuple=False)

# # Print the results
# for pair in high_colinearity_pairs:
#     latent1, latent2 = pair.tolist()
#     print(
#         f"Latent {sorted_latent_idxs[latent1]} and Latent {sorted_latent_idxs[latent2]} have colinearity {self_colinearity_matrix[latent1, latent2]} > {threshold}"
#     )

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=colinearity_matrix,
        x=probe_bases_labels,
        # y=sorted_latent_idxs.tolist(),
    )
)
fig.update_layout(title="Latent alignment with linear probes")
fig.show()

# Get the 10 highest absolute values in colinearity_matrix and their corresponding (y, x) pairs
top_k = 200
abs_colinearity = colinearity_matrix.abs().nan_to_num(0)
top_abs_values, top_indices = t.topk(abs_colinearity.flatten(), k=top_k)
top_values = colinearity_matrix.flatten()[top_indices]

# Convert flat indices to (y, x) pairs
top_yx_pairs = [
    (idx // abs_colinearity.shape[1], idx % abs_colinearity.shape[1])
    for idx in top_indices
]

# Print the results
for i, (value, (y, x)) in enumerate(zip(top_values, top_yx_pairs)):
    print(
        f"Value #{i}: {value.item()}, (y, x): ({sorted_latent_idxs[y]}, {probe_bases_labels[x]}), frac_active {frac_active[sorted_latent_idxs[y]]}"
    )


# %%
def visualise_dataset_activations(
    sae: OthelloSAE,
    latent_idx: int,
    dataset: Dataset | None = None,
    topk: int = 3,
):
    # Main display: text games
    # Hover/select: plot game
    if dataset is None:
        dataset = sae.test_dataset

    with t.inference_mode():
        forward_dict = sae.forward_dataset(dataset)

    acts_post = forward_dict["acts_post"].cpu()
    acts_post = acts_post.reshape(-1, sae.model.cfg.n_ctx, sae.cfg.d_sae)[
        ..., latent_idx
    ]

    # Sort acts_post and input_ids by the l0 norm in acts_post along dim -1
    l0_per_game = (acts_post.abs() > 1e-8).sum(1)
    l1_per_game = acts_post.abs().sum(1)
    sorted_indices = t.argsort(l0_per_game, dim=0, descending=True)[:topk]
    print(l0_per_game[sorted_indices], l1_per_game[sorted_indices])

    acts_post = acts_post[sorted_indices]
    dataset = datasets.concatenate_datasets(
        [Dataset.from_dict(d) for d in dataset]
    ).select(sorted_indices)

    print(sorted_indices.shape, acts_post.shape, len(dataset))

    for i, d in enumerate(dataset):
        plot_game(d)
        print(acts_post[i])


# latent_idx = sorted_latent_idxs[100].item()
latent_idx = 1336
latent_normed = sae.W_dec_normalized[latent_idx] @ post_matrix
latent_normed /= latent_normed.norm(dim=-1, keepdim=True)
# latent_normed = sae.W_enc.T[latent_idx]
cosine_similarity = (latent_normed @ probes_normed).detach().cpu()
print(latent_normed.shape, probes_normed.shape, cosine_similarity.shape)

# Create a DataFrame with the results
cosine_similarity_df = pd.DataFrame(
    cosine_similarity.numpy().reshape(len(colinear_keys), -1),
    index=colinear_keys,
    columns=[move_id_to_text(i, size) for i in actually_all_squares],
).T

print(latent_idx)
print(cosine_similarity_df.where(cosine_similarity_df.abs() > 0.1))
# print(cosine_similarity_df)
print(latent_idx)
visualise_dataset_activations(sae, latent_idx, topk=1)

# %%
w_ep = model.W_E_pos.clone()
# w_ep /= w_ep.norm(dim=-1, keepdim=True)
w_ep_basis = w_ep.clone()
# Perform Gram-Schmidt process to orthogonalize the rows of w_ep
for _ in tqdm(range(8)):
    for i in range(w_ep.shape[0]):
        for j in range(w_ep.shape[0]):
            if j == i:
                continue
            projection = (
                (w_ep_basis[i] @ w_ep_basis[j])
                / (w_ep_basis[j] @ w_ep_basis[j])
                * w_ep_basis[j]
            )
            w_ep_basis[i] -= projection

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=(w_ep_basis @ w_ep_basis.T > 0.1).float().cpu(),
    )
)
fig.show()

# %%
diag = t.diag(w_ep_basis @ w_ep_basis.T)
w_ep_basis[diag > 1e-6] /= diag[diag > 1e-6].sqrt().unsqueeze(-1)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=(w_ep_basis @ w_ep_basis.T).detach().cpu(),
    )
)
# w_ep (63, 64)
# basis (36, 64)

# %%
print(w_ep_basis.shape)
l0_sae = OthelloSAE(
    OthelloSAEConfig(
        d_sae=w_ep_basis.shape[0],
    ),
    model,
    train_dataset,
    test_dataset,
)

# %%
l0_sae.W_enc.data = w_ep_basis
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=(model.W_E_pos @ w_ep_basis.T).detach().cpu(),
    )
)
fig.show()
