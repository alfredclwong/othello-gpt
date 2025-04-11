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
from enum import Enum, auto


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


# hook_embed
# hook_pos_embed
# blocks.0.hook_resid_pre
# blocks.0.ln1.hook_scale
# blocks.0.ln1.hook_normalized
# blocks.0.attn.hook_q
# blocks.0.attn.hook_k
# blocks.0.attn.hook_v
# blocks.0.attn.hook_attn_scores
# blocks.0.attn.hook_pattern
# blocks.0.attn.hook_z
# blocks.0.hook_attn_out
# blocks.0.hook_resid_mid
# blocks.0.ln2.hook_scale
# blocks.0.ln2.hook_normalized
# blocks.0.mlp.hook_pre
# blocks.0.mlp.hook_post
# blocks.0.hook_mlp_out
# blocks.0.hook_resid_post
# ln_final.hook_scale
# ln_final.hook_normalized


@dataclass(frozen=True)
class SAEConfig:
    d_in: int
    d_sae: int
    in_hook_layer: int
    in_hook_suffix: str
    out_hook_layer: int
    out_hook_suffix: str

    sparsity_coeff: float = 1e-2
    downstream_coeff: float = 0

    n_epochs: int = 2
    n_steps_per_epoch: int = 14_000
    batch_size: int = 128

    log_steps: int = 1000
    log_warmup_steps: int = 100

    lr: float = 1e-3
    lr_scale: Callable[[int, int], float] = constant_lr
    lr_warmup_steps: int = 50
    lr_warmup_scale: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)

    resample_method: Literal["simple", "advanced", None] = "simple"
    resample_freq: int = 8000
    resample_window: int = 2000
    resample_scale: float = 0.5

    use_wandb: bool = True
    weight_normalize_eps: float = 1e-8
    architecture: Literal["standard", "gated", "jumprelu"] = "standard"

    use_b_dec: bool = True
    dead_threshold: float = 1e-8


class SAEType(Enum):
    LN_EMBED = auto()
    LN1 = auto()
    ATTN_Z = auto()
    LN2 = auto()
    TRANSCODER = auto()


class SAE(t.nn.Module, hf.PyTorchModelHubMixin):
    W_enc: Float[Tensor, "d_in d_sae"]
    W_dec: Float[Tensor, "d_sae d_in"]
    b_enc: Float[Tensor, "d_sae"]
    b_dec: Float[Tensor, "d_in"]

    def __init__(
        self,
        cfg: SAEConfig,
        model: HookedTransformer,
        device: str,
    ):
        super(SAE, self).__init__()

        self.cfg = cfg
        self.device = device
        self.model = model
        self.model.requires_grad_(False)
        self.in_hook_name = f"blocks.{cfg.in_hook_layer}.{cfg.in_hook_suffix}"
        self.out_hook_name = f"blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"

        self.post_matrix = (
            t.eye(cfg.d_in, device=device)
            if cfg.out_hook_suffix == "attn.hook_z"
            else model.W_O[cfg.out_hook_layer].flatten(0, 1)
        )

        downstream_weights = [
            model.W_Q[cfg.out_hook_layer + 1 :],
            model.W_K[cfg.out_hook_layer + 1 :],
            model.W_V[cfg.out_hook_layer + 1 :],
            model.W_in[
                cfg.out_hook_layer + int(cfg.out_hook_suffix == "hook_mlp_out") :
            ],
            model.W_U.unsqueeze(0),
        ]
        downstream_weights = t.cat(
            [w.transpose(-2, -1).flatten(0, -2) for w in downstream_weights], dim=0
        )
        downstream_weights /= downstream_weights.norm(dim=-1, keepdim=True)
        self.downstream_weights = downstream_weights

        self.W_enc = t.nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(self.cfg.d_in, self.cfg.d_sae))
        )
        self.W_dec = t.nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(self.cfg.d_sae, self.cfg.d_in))
        )
        self.W_dec.data = self.W_dec_normalized
        self.b_enc = t.nn.Parameter(t.zeros(self.cfg.d_sae))
        self.b_dec = t.zeros(self.cfg.d_in, device=device)
        if cfg.use_b_dec:
            self.b_dec = t.nn.Parameter(self.b_dec)

        self.to(device)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "d_sae d_in"]:
        return self.W_dec / (
            self.W_dec.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

    @property
    def sae_type(self) -> SAEType:
        if self.cfg.in_hook_layer == self.cfg.out_hook_layer:
            if self.in_hook_name == self.out_hook_name:
                if self.cfg.in_hook_suffix == "attn.hook_z":
                    return SAEType.ATTN_Z
                if self.in_hook_name == "blocks.0.ln1.hook_normalized":
                    return SAEType.LN_EMBED
            else:
                if self.cfg.in_hook_suffix == "ln2.hook_normalized" and self.cfg.out_hook_suffix == "hook_mlp_out":
                    return SAEType.TRANSCODER
                if self.cfg.in_hook_suffix == "hook_resid_pre" and self.cfg.out_hook_suffix == "ln1.hook_normalized":
                    return SAEType.LN1
        raise ValueError("Unrecognised SAE type", self)

    def forward(
        self,
        x_in: Float[Tensor, "batch d_in"],
        x_out: Float[Tensor, "batch d_in"],
    ) -> tuple[
        dict[str, Float[Tensor, "batch"]],
        dict[str, Float[Tensor, "batch d_sae"]],
        Float[Tensor, "batch d_in"],
    ]:
        x_c = x_in - self.b_dec
        acts_pre = x_c @ self.W_enc + self.b_enc
        acts_post = t.nn.functional.relu(acts_pre)
        x_recon = acts_post @ self.W_dec_normalized + self.b_dec

        l_recon = (x_out - x_recon).pow(2).mean(-1)
        l_sparsity = acts_post.abs().sum(-1)
        l_downstream = (
            -(x_recon @ self.post_matrix @ self.downstream_weights.T).abs().mean(-1)
        )
        # l_downstream = -(self.W_dec_normalized @ self.downstream_weights.T).abs().mean()
        # l_downstream = l_downstream.repeat(x_in.shape[0])
        l_sae = (
            l_recon
            + self.cfg.sparsity_coeff * l_sparsity
            + self.cfg.downstream_coeff * l_downstream
        )
        l_dict = {
            "L_recon": l_recon,
            "L_sparsity": l_sparsity,
            "L_downstream": l_downstream,
            "L_sae": l_sae,
        }
        a_dict = {
            "acts_pre": acts_pre,
            "acts_post": acts_post,
        }

        return l_dict, a_dict, x_recon

    def forward_dataset(self, dataset: Dataset, keys: list = []) -> dict[str, Tensor]:
        """
        Forward pass on a batched Dataset.
        """
        data = []

        for batch in dataset:
            with t.inference_mode():
                # TODO batch calculate activations and sample randomly to avoid training batches with tokens from the same game
                input_ids = t.tensor(batch["input_ids"], device=self.device)[:, :-1]
                names_filter = list(set([self.in_hook_name, self.out_hook_name]))
                stop_at_layer = max(self.cfg.in_hook_layer, self.cfg.out_hook_layer) + 1
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=names_filter,
                    stop_at_layer=stop_at_layer,
                )
                x_in: Float[Tensor, "(batch pos) d_model"] = (
                    cache[self.in_hook_name].flatten(2).flatten(0, 1)
                )
                x_out: Float[Tensor, "(batch pos) d_model"] = (
                    cache[self.out_hook_name].flatten(2).flatten(0, 1)
                )

            loss_dict, acts_dict, x_recon = self.forward(x_in, x_out)
            d = {
                "x_in": x_in,
                "x_out": x_out,
                "x_recon": x_recon,
                **loss_dict,
                **acts_dict,
            }

            if keys:
                d = {k: d[k] for k in d if k in keys}

            data.append(d)

        # Unbatch
        forward_dict = {k: t.cat([d[k] for d in data], dim=0) for k in data[0]}
        return forward_dict

    def optimise(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        if self.cfg.use_wandb:
            wandb.init(project="othello-gpt-sae", config=self.cfg)

        n_steps = self.cfg.n_epochs * self.cfg.n_steps_per_epoch
        pbar = tqdm(range(n_steps))

        batched_train_dataset = (
            train_dataset.select_columns(["input_ids"])
            .shuffle(seed=0)
            .take(min(len(train_dataset), n_steps * self.cfg.batch_size))
            .batch(self.cfg.batch_size)
        )
        batched_test_dataset = test_dataset.select_columns(["input_ids"]).batch(
            self.cfg.batch_size
        )

        optimizer = t.optim.Adam(
            list(self.parameters()), lr=self.cfg.lr, betas=self.cfg.betas
        )
        frac_active_in_window = deque(maxlen=self.cfg.resample_window)

        for step in pbar:
            # Update learning rate
            step_lr = self.cfg.lr * self.cfg.lr_scale(step, n_steps)
            if step % self.cfg.resample_freq < self.cfg.lr_warmup_steps:
                step_lr *= self.cfg.lr_warmup_scale
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            batch_dataset = batched_train_dataset.select(
                [step % len(batched_train_dataset)]
            )
            forward_dict = self.forward_dataset(batch_dataset)
            forward_dict["L_sae"].mean(0).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly
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
                eval_dict, _ = self.evaluate(batched_test_dataset)
                pbar.set_postfix(
                    {
                        "step": step,
                        "lr": step_lr,
                        **eval_dict,
                    }
                )
                if self.cfg.use_wandb:
                    wandb.log(
                        {
                            "lr": step_lr,
                            **eval_dict,
                        },
                        step=step,
                    )

        if self.cfg.use_wandb:
            wandb.finish()

    def evaluate(self, batched_dataset: Dataset):
        eval_dict = {}

        with t.inference_mode():
            # forward_keys = ["x_out", "acts_post", "x_recon"]
            forward_keys = []
            forward_dict = self.forward_dataset(batched_dataset, keys=forward_keys)
        eval_dict["x_norm"] = forward_dict["x_out"].norm(dim=-1).mean().item()

        input_ids = t.cat(
            [
                t.tensor(d["input_ids"], device=self.device)[:, :-1]
                for d in batched_dataset
            ],
            dim=0,
        )

        def zero_ablation_hook(x: Float[Tensor, "batch d_in"], hook: HookPoint):
            return t.zeros_like(x)

        def mean_ablation_hook(x: Float[Tensor, "batch d_in"], hook: HookPoint):
            return x.mean(dim=0, keepdim=True)

        def sae_hook(x: Float[Tensor, "batch d_in"], hook: HookPoint):
            return forward_dict["x_recon"].reshape_as(x)

        loss_dict = {
            k: v.mean().item() for k, v in forward_dict.items() if k.startswith("L_")
        }
        eval_dict.update(loss_dict)
        eval_dict["n_dead"] = (
            (forward_dict["acts_post"] < self.cfg.dead_threshold).all(0).sum(-1).item()
        )
        eval_dict["n_alive"] = self.cfg.d_sae - eval_dict["n_dead"]
        eval_dict["l0"] = (
            (forward_dict["acts_post"] >= self.cfg.dead_threshold)
            .float()
            .sum(-1)
            .mean()
            .item()
        )
        eval_dict["frac_active"] = eval_dict["l0"] / eval_dict["n_alive"]

        with t.inference_mode():
            self.model.reset_hooks()
            loss_zero = self.model.run_with_hooks(
                input_ids,
                return_type="loss",
                fwd_hooks=[(self.out_hook_name, zero_ablation_hook)],
            )

            self.model.reset_hooks()
            loss_mean = self.model.run_with_hooks(
                input_ids,
                return_type="loss",
                fwd_hooks=[(self.out_hook_name, mean_ablation_hook)],
            )

            self.model.reset_hooks()
            logits_recon, loss_recon = self.model.run_with_hooks(
                input_ids,
                return_type="both",
                fwd_hooks=[(self.out_hook_name, lambda x, hook: sae_hook(x, hook))],
            )

            self.model.reset_hooks()
            logits_original, loss_original = self.model(input_ids, return_type="both")

        loss_recovered_zero = (
            (1 - (loss_recon - loss_original) / (loss_zero - loss_original))
            .mean()
            .item()
        )
        loss_recovered_mean = (
            (1 - (loss_recon - loss_original) / (loss_mean - loss_original))
            .mean()
            .item()
        )
        kl_div = t.nn.functional.kl_div(
            logits_recon.log_softmax(dim=-1),
            logits_original.softmax(dim=-1),
            reduction="batchmean",
        )
        eval_dict["kl_div"] = kl_div.item()
        eval_dict["loss_recovered_zero_abl"] = loss_recovered_zero
        eval_dict["loss_recovered_mean_abl"] = loss_recovered_mean

        return eval_dict, forward_dict

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
        dead_neurons = (frac_active_in_window < self.cfg.dead_threshold).all(0)
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
        self.W_enc.data.T[dead_neurons] = resampled_neurons * resample_scale
        self.b_enc.data[dead_neurons] = 0
