# %%
from pathlib import Path

import torch as t
from datasets import Dataset, load_dataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jaxtyping import Float
from torch.types import Tensor
import einops

from othello_gpt.data.vis import move_id_to_text, plot_game
from othello_gpt.util import load_model, load_probes, get_all_squares
from othello_gpt.model.sae import SAE, SAEConfig
import datasets

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

root_dir = Path().cwd()#.parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

model_version = "600k"
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
cfg = SAEConfig(
    d_in=model.cfg.d_model,
    d_sae=1024,
    in_hook_layer=2,
    out_hook_layer=2,
    # in_hook_suffix="attn.hook_z",
    # out_hook_suffix="attn.hook_z",
    in_hook_suffix="ln2.hook_normalized",
    out_hook_suffix="hook_mlp_out",
)
is_z = "hook_z" in cfg.out_hook_suffix
in_hook_name = f"blocks.{cfg.in_hook_layer}.{cfg.in_hook_suffix}"
out_hook_name = f"blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"
sae_name = f"{model_name}-sae-{in_hook_name}"
if in_hook_name != out_hook_name:
    sae_name += f"-{out_hook_name}"
sae =SAE.from_pretrained(
    sae_name,
    cfg=cfg,
    model=model,
    device=device,
)

batched_test_dataset = test_dataset.take(1024).batch(128)
with t.inference_mode():
    eval_dict, test_forward_dict = sae.evaluate(batched_test_dataset)

acts_pre = test_forward_dict["acts_pre"].reshape(
    -1, sae.model.cfg.n_ctx, sae.cfg.d_sae
)
acts_post = test_forward_dict["acts_post"].reshape(
    -1, sae.model.cfg.n_ctx, sae.cfg.d_sae
)
frac_active = (acts_post > 1e-8).float().flatten(0, 1).mean(0)
sorted_latent_idxs = t.argsort(frac_active, dim=-1, descending=True)

{k: eval_dict[k] for k in sorted(eval_dict.keys())}

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
    [probes[k][..., sae.cfg.out_hook_layer] for k in probe_keys], dim=1
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
x = test_forward_dict["x_out"]
x_recon = test_forward_dict["x_recon"]
err = x - x_recon
[v.norm(dim=-1).mean().item() for v in [x, x_recon, err]], test_forward_dict["acts_post"].norm(0, dim=-1).mean()

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Activation heatmap", "Frac active"],
)
fig.add_trace(
    go.Heatmap(
        z=acts_post[:10, :, sorted_latent_idxs].flatten(0, 1).cpu(),
        showscale=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(y=frac_active[sorted_latent_idxs].cpu(), showlegend=False),
    row=2,
    col=1,
)
fig.show()

# %%
latents_normed: Float[Tensor, "d_sae d_in"] = sae.W_dec_normalized.detach().clone()
latents_normed /= latents_normed.norm(dim=-1, keepdim=True)
latents_normed = latents_normed[sorted_latent_idxs]
if is_z:
    latents_normed = latents_normed.reshape(sae.cfg.d_sae, sae.model.cfg.n_heads, sae.model.cfg.d_head)
    w_o_normed = sae.model.W_O[sae.cfg.out_hook_layer]
    w_o_normed /= w_o_normed.norm(dim=-1, keepdim=True)
    latents_normed = einops.einsum(
        latents_normed,
        w_o_normed,
        "d_sae n_head d_head, n_head d_head d_model -> d_sae n_head d_model"
    ).flatten(0, 1)

self_colinearity_matrix = latents_normed @ latents_normed.T
self_colinearity_matrix = t.tril(self_colinearity_matrix, diagonal=-1)
colinearity_matrix = latents_normed @ probes_normed
flat_colinearity_matrix = colinearity_matrix.flatten(0, 1)

# Generate random vectors for comparison
random_vectors = t.randn_like(latents_normed)
random_vectors /= random_vectors.norm(dim=-1, keepdim=True)
random_colinearity_matrix = (random_vectors @ random_vectors.T).abs()
random_colinearity_matrix = t.tril(random_colinearity_matrix, diagonal=-1)

# Calculate the maximum values along the 0th dimension
self_max_values = self_colinearity_matrix.max(0)[0].cpu().numpy()
max_values = flat_colinearity_matrix.max(0)[0].cpu().numpy()
random_max_values = random_colinearity_matrix.max(0)[0].cpu().numpy()

# %%
top_k = sae.cfg.d_sae
abs_colinearity = colinearity_matrix.abs().nan_to_num(0)
_, top_indices = t.topk(abs_colinearity.flatten(), k=top_k)
for k, idx in enumerate(top_indices):
    colinearity = colinearity_matrix.flatten()[idx].item()
    latent_idx, probe_idx = divmod(idx.item(), colinearity_matrix.shape[-1])
    lh_idx = divmod(latent_idx, sae.model.cfg.n_heads) if is_z else (latent_idx, None)
    pct_active = frac_active[sorted_latent_idxs[lh_idx[0]]]
    print(
        f"{sae.out_hook_name} #{k}: {colinearity=:.2f}, {lh_idx=}, {probe_bases_labels[probe_idx]}, {pct_active=:.2%}"
    )

# %%
# Given (sae_idx, latent_idx), find the max activating datasets and plot the games with activation values
top_k = 3
sorted_latent_idx = 122

heads = list(range(sae.model.cfg.n_heads)) if is_z else [0]
for h in heads:
    latent_normed = latents_normed[h + sorted_latent_idx * len(heads)]
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
    display(latent_probe_df)

latent_idx = sorted_latent_idxs[sorted_latent_idx]
max_act_per_game = acts_post[..., latent_idx].max(dim=1)[0]
_, top_game_idxs = t.topk(max_act_per_game, k=top_k)
flat_dataset = datasets.concatenate_datasets([Dataset.from_dict(d) for d in batched_test_dataset])
for game_idx in top_game_idxs.tolist():
    game_acts = acts_pre[game_idx, :, latent_idx]
    plot_game(
        flat_dataset[game_idx],
        subplot_titles=[
            f"<b style='color:red;'>{act:.2f}</b>" if act > 0 else f"{act:.2f}"
            for act in game_acts.tolist()
        ],
    )

# %%
# DFA: Direct Feature Attribution
# Each z can be split into n_head parts
# Showed above that each part can be out projected and probed. A good (sparse) feature
# should only show alignment from one head to the linear probe basis.
# DFA uses the cached A and V values to decompose z into source token contributions
# These can in turn be decomposed into upstream latents
game = test_dataset[0]
plot_game(game)
