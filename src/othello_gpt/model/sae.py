import numpy as np
from dataclasses import dataclass
from typing import Literal, Callable, Tuple, Any, List
from collections import deque
from jaxtyping import Float
from torch.types import Tensor
import torch as t
import huggingface_hub as hf
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import wandb
from tqdm import tqdm
import einops
from itertools import product
import pandas as pd


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass(frozen=True)
class OthelloSAEConfig:
    d_in: int
    d_sae: int
    hook_layers: List[int]
    hook_suffixes: List[str]
    # Examples:
    # 'hook_embed',
    # 'hook_pos_embed',
    # 'blocks.0.hook_resid_pre',
    # 'blocks.0.ln1.hook_scale',
    # 'blocks.0.ln1.hook_normalized',
    # 'blocks.0.attn.hook_q',
    # 'blocks.0.attn.hook_k',
    # 'blocks.0.attn.hook_v',
    # 'blocks.0.attn.hook_attn_scores',
    # 'blocks.0.attn.hook_pattern',
    # 'blocks.0.attn.hook_z',
    # 'blocks.0.hook_attn_out',
    # 'blocks.0.hook_resid_mid',
    # 'blocks.0.ln2.hook_scale',
    # 'blocks.0.ln2.hook_normalized',
    # 'blocks.0.mlp.hook_pre',
    # 'blocks.0.mlp.hook_post',
    # 'blocks.0.hook_mlp_out',
    # 'blocks.0.hook_resid_post',
    # 'ln_final.hook_scale',
    # 'ln_final.hook_normalized'

    use_wandb: bool = True
    sparsity_coeff: float = 0.1
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated", "jumprelu"] = "standard"

    n_train: int = 1_792_000
    n_test: int = 1024
    batch_size: int = 128  # 14_000 steps
    n_epochs: int = 8

    log_steps: int = 1000
    log_warmup_steps: int = 100

    lr: float = 1e-3
    lr_scale: Callable[[int, int], float] = cosine_decay_lr
    lr_warmup_steps: int = 50
    lr_warmup_scale: float = 0.1

    resample_method: Literal["simple", "advanced", None] = "simple"
    resample_freq: int = 14_000
    resample_window: int = 2000
    resample_scale: float = 0.5

    betas: Tuple[float, float] = (0.9, 0.999)


class OthelloSAE(t.nn.Module, hf.PyTorchModelHubMixin):
    W_enc: Float[Tensor, "n_sae d_in d_sae"]
    _W_dec: Float[Tensor, "n_sae d_sae d_in"] | None
    b_enc: Float[Tensor, "n_sae d_sae"]
    b_dec: Float[Tensor, "n_sae d_in"]

    def __init__(
        self,
        sae_cfg: OthelloSAEConfig,
        model: HookedTransformer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        device: str,
    ):
        super(OthelloSAE, self).__init__()

        self.cfg = sae_cfg
        self.device = device
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

        self.hook_names = [
            f"blocks.{hook_layer}.{hook_suffix}"
            for hook_layer, hook_suffix in product(
                self.cfg.hook_layers, self.cfg.hook_suffixes
            )
        ]
        self.n_sae = len(self.hook_names)
        self.W_enc = t.nn.Parameter(
            t.nn.init.kaiming_uniform_(
                t.empty(self.n_sae, self.cfg.d_in, self.cfg.d_sae)
            )
        )
        if self.cfg.tied_weights:
            self.W_dec = None
        else:
            self.W_dec = t.nn.Parameter(
                t.nn.init.kaiming_uniform_(
                    t.empty(self.n_sae, self.cfg.d_sae, self.cfg.d_in)
                )
            )
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = t.nn.Parameter(t.zeros(self.n_sae, self.cfg.d_sae))
        self.b_dec = t.nn.Parameter(t.zeros(self.n_sae, self.cfg.d_in))

        self.to(self.device)

    @property
    def W_dec(self) -> Float[Tensor, "n_sae d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-2, -1)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "n_sae d_sae d_in"]:
        return self.W_dec / (
            self.W_dec.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

    def evaluate(self):
        eval_dict = {}

        test_forward_dict = self.forward_dataset(self.test_dataset)
        dead_latent_count = (test_forward_dict["acts_post"] < 1e-8).all(0).sum(-1)
        frac_active = (test_forward_dict["acts_post"] >= 1e-8).float().mean(0).mean(-1)
        test_input_ids = t.cat(
            [
                t.tensor(d["input_ids"], device=self.device)[:, :-1]
                for d in self.test_dataset
            ],
            dim=0,
        )

        def zero_ablation_hook(x, hook: HookPoint):
            return t.zeros_like(x)

        def sae_hook(x, hook: HookPoint, sae_index: int):
            return test_forward_dict["x_recon"][:, sae_index].reshape_as(x)

        for i, hook_name in enumerate(self.hook_names):
            loss_dict = {
                f"{k}_{hook_name}": v[:, i].mean(0).item()
                for k, v in test_forward_dict.items()
                if k.startswith("L_")
            }
            eval_dict.update(loss_dict)
            eval_dict[f"n_dead_{hook_name}"] = dead_latent_count[i].item()
            eval_dict[f"frac_active_{hook_name}"] = frac_active[i].item()

            with t.inference_mode():
                self.model.reset_hooks()
                loss_recon = self.model.run_with_hooks(
                    test_input_ids,
                    return_type="loss",
                    fwd_hooks=[(hook_name, lambda x, hook: sae_hook(x, hook, i))],
                )

                self.model.reset_hooks()
                loss_zero = self.model.run_with_hooks(
                    test_input_ids,
                    return_type="loss",
                    fwd_hooks=[(hook_name, zero_ablation_hook)],
                )

                self.model.reset_hooks()
                loss_original = self.model(test_input_ids, return_type="loss")

            loss_recovered_zero = (
                (1 - (loss_recon - loss_original) / (loss_zero - loss_original))
                .mean()
                .item()
            )
            eval_dict[f"loss_recovered_{hook_name}"] = loss_recovered_zero

        return eval_dict

    def forward(
        self, x: Float[Tensor, "batch n_sae d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch n_sae"]],
        Float[Tensor, "batch n_sae d_sae"],
        Float[Tensor, "batch n_sae d_in"],
    ]:
        x_c = x - self.b_dec
        acts_pre = (
            einops.einsum(
                x_c,
                self.W_enc,
                "batch n_sae d_in, n_sae d_in d_sae -> batch n_sae d_sae",
            )
            + self.b_enc
        )
        acts_post = t.nn.functional.relu(acts_pre)
        x_recon = (
            einops.einsum(
                acts_post,
                self.W_dec_normalized,
                "batch n_sae d_sae, n_sae d_sae d_in -> batch n_sae d_in",
            )
            + self.b_dec
        )

        l_recon = (x - x_recon).pow(2).mean(-1)  # avg for mse
        l_recon /= x.norm(dim=-1).mean(0, keepdim=True)
        l_sparsity = acts_post.abs().sum(-1)
        # sum because ground truth number of features is indep of d_sae
        l_sae = l_recon + self.cfg.sparsity_coeff * l_sparsity
        l_dict = {"L_recon": l_recon, "L_sparsity": l_sparsity, "L_sae": l_sae}

        return l_dict, acts_post, x_recon

    def forward_dataset(
        self, dataset: Dataset, include_weights: bool = False
    ) -> dict[str, Tensor]:
        """
        Forward pass on a batched Dataset.
        """
        data = []

        for batch in dataset:
            with t.inference_mode():
                # TODO batch calculate activations and sample randomly to avoid training batches with tokens from the same game
                input_ids = t.tensor(batch["input_ids"], device=self.device)[:, :-1]
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=self.hook_names,
                    stop_at_layer=max(self.cfg.hook_layers) + 1,
                )
                x: Float[Tensor, "(batch pos) n_sae d_model"] = t.stack(
                    [cache[hook_name].flatten(2) for hook_name in self.hook_names],
                    dim=2,
                ).flatten(0, 1)  # flatten(2) for attn_z

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

        if include_weights:
            forward_dict |= {
                name: param.detach().cpu()
                for name, param in self.named_parameters(recurse=False)
            }

        return forward_dict

    def optimize(self, log_data: bool = False) -> list[dict[str, Any]]:
        # TODO separate into a trainer class/func

        assert self.cfg.resample_window <= self.cfg.resample_freq

        if self.cfg.use_wandb:
            wandb.init(project="othello-gpt-sae", config=self.cfg)

        optimizer = t.optim.Adam(
            list(self.parameters()), lr=self.cfg.lr, betas=self.cfg.betas
        )
        n_steps = self.cfg.n_epochs * self.cfg.n_train // self.cfg.batch_size
        progress_bar = tqdm(range(n_steps))

        # Create lists of dicts to store data we'll eventually be plotting
        data_log = []
        frac_active_in_window = deque(maxlen=self.cfg.resample_window)

        for step in progress_bar:
            # Update learning rate
            step_lr = self.cfg.lr * self.cfg.lr_scale(step, n_steps)
            if step % self.cfg.resample_freq < self.cfg.lr_warmup_steps:
                step_lr *= self.cfg.lr_warmup_scale
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            step_dataset = self.train_dataset.select([step % len(self.train_dataset)])
            forward_dict = self.forward_dataset(step_dataset)
            forward_dict["L_sae"].mean(0).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized.data

            # Calculate the mean sparsities over batch dim for each feature
            active = forward_dict["acts_post"] > 1e-8
            frac_active = active.float().mean(0)
            frac_active_in_window.append(frac_active)

            # Resample dead latents
            resample = (
                (self.cfg.resample_method is not None)
                and ((step + 1) % self.cfg.resample_freq == 0)
                and (step + 1 + self.cfg.resample_freq < n_steps)
            )
            if resample:
                if self.cfg.resample_method == "simple":
                    self.resample_simple(
                        t.stack(list(frac_active_in_window), dim=0),
                        self.cfg.resample_scale,
                    )
                elif self.cfg.resample_method == "advanced":
                    raise NotImplementedError

            # Display progress bar, and log a bunch of values for creating plots / animations
            if not (step % self.cfg.resample_freq < self.cfg.log_warmup_steps) and (
                step % self.cfg.log_steps == 0 or (step + 1 == n_steps)
            ):
                eval_dict = self.evaluate()
                eval_groups = [
                    "L_recon",
                    "L_sparsity",
                    "L_sae",
                    "n_dead",
                    "loss_recovered",
                ]
                avg_eval_dict = {
                    k: sum(
                        eval_dict[f"{k}_{hook_name}"] for hook_name in self.hook_names
                    )
                    / len(self.hook_names)
                    for k in eval_groups
                }

                progress_bar.set_postfix(
                    {
                        "step": step,
                        "lr": step_lr,
                        **avg_eval_dict,
                    }
                )

                if log_data:
                    data_log.append(eval_dict | avg_eval_dict)

                if self.cfg.use_wandb:
                    wandb.log(
                        {"lr": step_lr, **eval_dict},
                        step=step,
                    )

        if self.cfg.use_wandb:
            wandb.finish()

        return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window n_sae d_sae"],
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
        print(
            f"Resampling {dead_neurons.sum().item()}/{dead_neurons.numel()} dead neurons..."
        )
        resampled_neurons = t.randn(
            dead_neurons.sum(), self.cfg.d_in, device=self.W_dec.device
        )
        resampled_neurons /= (
            resampled_neurons.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )
        self.W_dec.data[dead_neurons] = resampled_neurons
        self.W_enc.data.transpose(-2, -1)[dead_neurons] = (
            resampled_neurons * resample_scale
        )
        self.b_enc.data[dead_neurons] = 0
