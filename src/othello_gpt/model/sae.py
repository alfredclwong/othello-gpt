# %%
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Tuple

import einops
import huggingface_hub as hf
import numpy as np
import torch as t
import torch.utils as utils
import wandb
from datasets import Dataset, load_dataset
from jaxtyping import Float
from torch.types import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from plotly.subplots import make_subplots

from othello_gpt.util import load_model

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)
device

# %%
# We start by emptying memory of all large tensors & objects (since we'll be loading in a lot of different models in the coming sections)
THRESHOLD = 0.1  # GB
for obj in gc.get_objects():
    try:
        if (
            isinstance(obj, t.nn.Module)
            and utils.get_tensors_size(obj) / 1024**3 > THRESHOLD
        ):
            if hasattr(obj, device):
                obj.cpu()
            if hasattr(obj, "reset"):
                obj.reset()
    except:
        pass

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
probe_dir.mkdir(parents=True, exist_ok=True)

hf.login((root_dir / "secret.txt").read_text())
wandb.login()

model_version = "300k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)

dataset_dict = load_dataset("awonga/othello-gpt")
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]


# %%
def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


# %%
# ActivationCache with keys [
#     'hook_embed',
#     'hook_pos_embed',
#     'blocks.0.hook_resid_pre',
#     'blocks.0.ln1.hook_scale',
#     'blocks.0.ln1.hook_normalized',
#     'blocks.0.attn.hook_q',
#     'blocks.0.attn.hook_k',
#     'blocks.0.attn.hook_v',
#     'blocks.0.attn.hook_attn_scores',
#     'blocks.0.attn.hook_pattern',
#     'blocks.0.attn.hook_z',
#     'blocks.0.hook_attn_out',
#     'blocks.0.hook_resid_mid',
#     'blocks.0.ln2.hook_scale',
#     'blocks.0.ln2.hook_normalized',
#     'blocks.0.mlp.hook_pre',
#     'blocks.0.mlp.hook_post',
#     'blocks.0.hook_mlp_out',
#     'blocks.0.hook_resid_post',
#     'ln_final.hook_scale',
#     'ln_final.hook_normalized'
# ])


# %%
@dataclass(frozen=True)
class OthelloSAEConfig:
    use_wandb: bool = True
    hook_name: str = "blocks.0.ln1.hook_normalized"
    hook_layer: int = 0

    d_in: int = model.cfg.d_model
    d_sae: int = model.cfg.d_model * 16
    sparsity_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated", "jumprelu"] = "standard"

    """
    batch_size:         size of batches we pass through model & train autoencoder on
    epochs:             number of optimization epochs
    log_freq:           number of optimization steps between logging
    lr:                 learning rate
    lr_scale:           learning rate scaling function
    resample_method:    method for resampling dead latents
    resample_freq:      number of optimization steps between resampling dead latents
    resample_window:    number of steps needed for us to classify a neuron as dead
    resample_scale:     scale factor for resampled neurons
    """
    n_train: int = 1_792_000
    n_test: int = 1024
    batch_size: int = 128  # 14_000 steps
    n_epochs: int = 5

    log_steps: int = 1000
    log_warmup_steps: int = 10

    lr: float = 2e-3
    lr_scale: Callable[[int, int], float] = cosine_decay_lr
    lr_warmup_steps: int = 50
    lr_warmup_scale: float = 0.1

    resample_method: Literal["simple", "advanced", None] = "simple"
    resample_freq: int = 14_000
    resample_window: int = 2000
    resample_scale: float = 0.5

    betas: Tuple[float, float] = (0.9, 0.999)


class OthelloSAE(t.nn.Module, hf.PyTorchModelHubMixin):
    # TODO all layers at once

    W_enc: Float[Tensor, "d_in d_sae"]
    _W_dec: Float[Tensor, "d_sae d_in"] | None
    b_enc: Float[Tensor, "d_sae"]
    b_dec: Float[Tensor, "d_in"]

    def __init__(
        self,
        sae_cfg: OthelloSAEConfig,
        model: HookedTransformer,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super(OthelloSAE, self).__init__()

        assert sae_cfg.d_in == model.cfg.d_model, (
            f"{sae_cfg.d_in=} != {model.cfg.d_model=}"
        )
        self.cfg = sae_cfg
        self.model = model
        self.model.requires_grad_(False)

        self.train_dataset = (
            train_dataset.shuffle(seed=0)
            .take(min(len(train_dataset), self.cfg.n_train * self.cfg.n_epochs))
            .select_columns(["input_ids"])
            .batch(self.cfg.batch_size)
        )
        self.test_dataset = (
            test_dataset.shuffle(seed=0)
            .take(self.cfg.n_test)
            .select_columns(["input_ids", "boards", "legalities", "moves"])
            .batch(self.cfg.batch_size)
        )

        self.W_enc = t.nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(self.cfg.d_in, self.cfg.d_sae))
        )
        if self.cfg.tied_weights:
            self.W_dec = None
        else:
            self.W_dec = t.nn.Parameter(
                t.nn.init.kaiming_uniform_(t.empty(self.cfg.d_sae, self.cfg.d_in))
            )
        self.b_enc = t.nn.Parameter(t.zeros(self.cfg.d_sae))
        self.b_dec = t.nn.Parameter(t.zeros(self.cfg.d_in))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.T

    @property
    def W_dec_normalized(self) -> Float[Tensor, "d_sae d_in"]:
        """
        Returns decoder weights, normalized over the autoencoder input dimension.
        """
        return self.W_dec / (
            self.W_dec.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

    def forward(
        self, x: Float[Tensor, "batch d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, ""]],
        Float[Tensor, "batch d_sae"],
        Float[Tensor, "batch d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            x:         activations of model

        Returns:
            l_dict:    dict of different loss terms, each dict value having shape (batch_size)
            l:         total loss (i.e. sum over terms of loss dict), same shape as loss_dict values
            acts_post: autoencoder latent activations, after applying ReLU
            x_r:       reconstructed autoencoder input
        """
        # x_n = x / x.norm(dim=-1, keepdim=True)
        x_c = x - self.b_dec
        acts_pre = x_c @ self.W_enc + self.b_enc
        acts_post = t.nn.functional.relu(acts_pre)
        x_recon = acts_post @ self.W_dec_normalized + self.b_dec

        l_recon = t.square(x - x_recon).sum(-1)  # avg over d_in to normalise
        l_sparsity = acts_post.abs().sum(-1)  # L1 norm
        l_sae = l_recon + self.cfg.sparsity_coeff * l_sparsity
        l_dict = {"L_recon": l_recon, "L_sparsity": l_sparsity, "L_sae": l_sae}

        # TODO model loss with x_recon patched

        return (l_dict, acts_post, x_recon)  # TODO store acts_pre?

    def forward_dataset(
        self, dataset: Dataset, include_weights: bool = False
    ) -> dict[str, Tensor]:
        """
        Forward pass on a batched Dataset.
        """
        data = []

        for batch in dataset:
            with t.inference_mode():
                input_ids = t.tensor(batch["input_ids"], device=device)[:, :-1]
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=self.cfg.hook_name,
                    stop_at_layer=self.cfg.hook_layer + 1,
                )
                # x: Float[Tensor, "(batch pos) d_model"] = cache.apply_ln_to_stack(
                x: Float[Tensor, "(batch pos) d_model"] = cache[
                    self.cfg.hook_name
                ].flatten(0, 1)
                # TODO ok to train on all pos?

            loss_dict, acts_post, x_recon = self.forward(x)
            data.append(
                {
                    **{name: loss_term for name, loss_term in loss_dict.items()},
                    "acts_post": acts_post,
                    "x_recon": x_recon,
                    "x": x,
                }
            )

        forward_dict = {k: t.cat([data[k] for data in data], dim=0) for k in data[0]}
        forward_dict = {
            k: v.mean(0) if k.startswith("L_") else v for k, v in forward_dict.items()
        }

        if include_weights:
            forward_dict |= {
                name: param.detach().cpu()
                for name, param in self.named_parameters(recurse=False)
            }

        return forward_dict

    def optimize(self) -> list[dict[str, Any]]:
        # TODO separate into a trainer class/func

        assert self.cfg.resample_window <= self.cfg.resample_freq

        if self.cfg.use_wandb:
            wandb.init(project="othello-gpt-sae", config=self.cfg)

        optimizer = t.optim.Adam(
            list(self.parameters()), lr=self.cfg.lr, betas=self.cfg.betas
        )
        n_steps = self.cfg.n_epochs * len(self.train_dataset)
        progress_bar = tqdm(range(n_steps))

        # Create lists of dicts to store data we'll eventually be plotting
        data_log = []
        frac_active_in_window = deque(maxlen=self.cfg.resample_window)

        for step in progress_bar:
            # Resample dead latents
            if (
                (self.cfg.resample_method is not None)
                and ((step + 1) % self.cfg.resample_freq == 0)
                and (step + 1 + self.cfg.resample_freq < n_steps)
            ):
                if self.cfg.resample_method == "simple":
                    self.resample_simple(
                        t.stack(list(frac_active_in_window), dim=0),
                        self.cfg.resample_scale,
                    )
                elif self.cfg.resample_method == "advanced":
                    raise NotImplementedError

            # Update learning rate
            step_lr = self.cfg.lr * self.cfg.lr_scale(step, n_steps)
            if step % self.cfg.resample_freq < self.cfg.lr_warmup_steps:
                step_lr *= self.cfg.lr_warmup_scale
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            step_dataset = self.train_dataset.select([step % len(self.train_dataset)])
            forward_dict = self.forward_dataset(step_dataset)
            forward_dict["L_sae"].backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized.data

            # Calculate the mean sparsities over batch dim for each feature
            # TODO is abs necessary for post-relu?
            frac_active = (forward_dict["acts_post"].abs() > 1e-8).float().mean(0)
            frac_active_in_window.append(frac_active)

            # Display progress bar, and log a bunch of values for creating plots / animations
            if step % self.cfg.log_steps == 0 or (step + 1 == n_steps):
                progress_bar.set_postfix(
                    step=step,
                    lr=step_lr,
                    frac_active=frac_active.mean().item(),
                    **{
                        k: v.mean().item()
                        for k, v in forward_dict.items()
                        if k.startswith("L_")
                    },
                )

                if step < self.cfg.log_warmup_steps:
                    continue

                with t.inference_mode():
                    test_forward_dict = self.forward_dataset(self.test_dataset)
                test_forward_dict = {k: v.cpu() for k, v in test_forward_dict.items()}
                data_log.append(test_forward_dict)

                if self.cfg.use_wandb:
                    # TODO acts_post histogram
                    # TODO W_dec similarity to feature set
                    test_frac_active = (
                        (test_forward_dict["acts_post"].abs() > 1e-8).float().mean(0)
                    )
                    frac_active_list = [[x] for x in test_frac_active.tolist()]
                    frac_active_table = wandb.Table(
                        data=frac_active_list, columns=["frac_active"]
                    )
                    frac_active_hist = wandb.plot.histogram(
                        frac_active_table, "frac_active"
                    )
                    wandb.log(
                        {
                            "lr": step_lr,
                            "frac_active": frac_active.mean().item(),
                            "frac_active_histogram": frac_active_hist,
                            **{
                                k: v.item()
                                for k, v in test_forward_dict.items()
                                if k.startswith("L_")
                            },
                        },
                        step=step,
                    )

        if self.cfg.use_wandb:
            wandb.finish()

        return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        dead_neurons = (frac_active_in_window < 1e-8).all(0)
        dead_neurons_count = dead_neurons.sum().item()
        print(
            f"Resampling {dead_neurons_count}/{dead_neurons.shape[0]} dead neurons..."
        )
        resampled_neurons = t.randn(
            dead_neurons.sum(), self.cfg.d_in, device=self.W_dec.device
        )
        resampled_neurons /= (
            resampled_neurons.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )
        self.W_dec.data[dead_neurons] = resampled_neurons
        self.W_enc.data.T[dead_neurons] = resampled_neurons * resample_scale
        self.b_enc.data[dead_neurons] = 0


# %%
for i in range(0, model.cfg.n_layers):
    sae_cfg = OthelloSAEConfig(
        use_wandb=True,
        hook_name=f"blocks.{i}.ln1.hook_normalized",
        hook_layer=i,
        d_sae=512 if i == 0 else 1024,
    )
    print(sae_cfg)
    sae = OthelloSAE(sae_cfg, model, train_dataset, test_dataset)
    data_log = sae.optimize()
    sae.push_to_hub(f"{model_name}-sae-{sae_cfg.hook_name}")

# %%
# Visualise:
#  - latent density histogram
#  - similarity to existing features
#  - max activating datasets
sae_layer = 0
sae_cfg = OthelloSAEConfig(
    hook_name=f"blocks.{sae_layer}.ln1.hook_normalized",
    hook_layer=sae_layer,
    d_sae=512 if sae_layer == 0 else 1024,
)
sae = OthelloSAE.from_pretrained(
    f"{model_name}-sae-{sae_cfg.hook_name}",
    sae_cfg=sae_cfg,
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
)

with t.inference_mode():
    test_forward_dict = sae.forward_dataset(sae.train_dataset.take(64))

# %%
x_norm = test_forward_dict["x"].norm(dim=-1).mean().item()
x_recon_norm = test_forward_dict["x_recon"].norm(dim=-1).mean().item()
x_recon_error = (test_forward_dict["x"] - test_forward_dict["x_recon"]).norm(dim=-1).mean().item()

print(f"x norm: {x_norm}")
print(f"x_recon norm: {x_recon_norm}")
print(f"x_recon error: {x_recon_error}")

# %%
acts_post_sample = (
    test_forward_dict["acts_post"]
    .reshape(-1, sae.model.cfg.n_ctx, sae.cfg.d_sae)
    .cpu()
    .clone()
)
print(acts_post_sample.shape)
frac_active = (acts_post_sample.abs() > 1e-8).float().flatten(0, 1).mean(0)
sorted_latent_idxs = t.argsort(frac_active, descending=True)
sorted_latent_idxs = sorted_latent_idxs[:t.argmax((frac_active[sorted_latent_idxs] == 0).float()).item()]
print(sorted_latent_idxs)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.05,
    subplot_titles=("Heatmap", "Fraction Active Scatter Plot"),
)

fig.add_trace(
    go.Heatmap(
        z=acts_post_sample[:100, :, sorted_latent_idxs].flatten(0, 1),
        # z=(
        #     acts_post_sample[:100, :, sorted_latent_idxs].flatten(0, 1).abs() > 1e-8
        # ).float(),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        y=frac_active[sorted_latent_idxs],
    ),
    row=2,
    col=1,
)

fig.show()

# %%
import datasets
from othello_gpt.data.vis import plot_game


def visualise_dataset_activations(
    sae: OthelloSAE, latent_idx: int, dataset: Dataset | None = None
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
    k = 3
    sorted_indices = t.argsort(l0_per_game, dim=0, descending=True)[:k]
    print(l0_per_game[sorted_indices], l1_per_game[sorted_indices])

    acts_post = acts_post[sorted_indices]
    dataset = datasets.concatenate_datasets(
        [Dataset.from_dict(d) for d in dataset]
    ).select(sorted_indices)

    print(sorted_indices.shape, acts_post.shape, len(dataset))

    for i, d in enumerate(dataset):
        plot_game(d)
        print(acts_post[i])


latent_idx = sorted_latent_idxs[50].item()
print(latent_idx)
visualise_dataset_activations(sae, latent_idx)

# %%
# Layer 5, latent 1950: E5/E3/E4 moves?
# Layer 1, latent 1878: even moves, B4 first move, excluding last ~6 moves
# Layer 1, latent 1938: odd moves, excluding first/last
# Layer 1, latent 32: even moves, excluding first 5 moves
# Layer 1, latent 730: E3 was the first move, first half, odd moves
# Layer 1, latent 153: D2 was the first move, first half, odd moves
# Layer 1, latent 1651: B4 was the first move
# Layer 1, latent 1054: last move, last ~5 moves

# Layer 0, latent 762: move 21, E1/F1/F6/F5
# Primariliy move 21, F1. weak activations on other moves (discrete categorisation)
# Layer 0, latent 366: F1, move 6


# %%
# Notes so far:
#   - Layer 0
#       - reconstruction perfect
#       - expect 32 (mov) * 4 (pos) = 128 features
#       - unit freq = 1 / 32 / 31 = 1e-3
#           - 44th-87th L0-sorted features are around the target freq and are interpretable!
#           - these represent unfactored, atomic features
#           - factorised representation is 36-dim vs 128-dim
#           - 242 alive latents, probably ~(32 moves + 32 TM (no corners)) * 4 pos

# %%
