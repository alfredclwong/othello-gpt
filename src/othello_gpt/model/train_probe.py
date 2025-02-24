# %%
import datetime as dt
from pathlib import Path
from typing import Callable, List

import huggingface_hub as hf
import numpy as np
import torch as t
import wandb
from datasets import load_dataset
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens import HookedTransformer

from othello_gpt.research.targets import (
    captures_target,
    flip_dir_target,
    forward_probe,
    legality_target,
    prev_tem_target,
    theirs_empty_mine_target,
)
from othello_gpt.util import (
    LinearProbeTrainingArgs,
    get_all_squares,
    load_model,
    test_linear_probe,
)

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
probe_dir.mkdir(parents=True, exist_ok=True)

# hf.login((root_dir / "secret.txt").read_text())
wandb.login()

size = 6
all_squares = get_all_squares(size)
dataset_dict = load_dataset("awonga/othello-gpt")

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# %%
def train_linear_probe(
    model: HookedTransformer,
    args: LinearProbeTrainingArgs,
    target_fn: Callable,
    target_cols: List[str] = [],
):
    cols = ["input_ids"] + target_cols
    test_dataset = dataset_dict["test"].take(args.n_test).select_columns(cols)
    test_y: Float[t.Tensor, "n_test pos n_out"] = target_fn(test_dataset)
    test_y = test_y.to(device)
    d_probe = test_y.max().item() + 1
    n_probes = model.cfg.n_layers * 2 + 1

    linear_probe = t.randn(
        (model.cfg.d_model, test_y.shape[-1], d_probe, n_probes)
    ) / np.sqrt(model.cfg.d_model)
    linear_probe = linear_probe.to(device)
    linear_probe.requires_grad = True
    print(f"{linear_probe.shape=}")

    test_loss, test_accs = test_linear_probe(
        model, device, test_dataset, test_y, linear_probe, target_fn
    )

    train_dataset = dataset_dict["train"]
    batch_indices = t.randint(
        0,
        len(train_dataset),
        (args.n_epochs, args.n_steps_per_epoch, args.batch_size),
    )

    optimizer = t.optim.AdamW(
        [linear_probe], lr=args.lr, betas=args.betas, weight_decay=args.weight_decay
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    pbar = tqdm(total=args.n_steps_per_epoch * args.n_epochs)
    step = 0
    for i in range(args.n_epochs):
        for j in range(args.n_steps_per_epoch):
            # TODO enable pin_memory
            batch = train_dataset.select(batch_indices[i, j, :])
            _, loss = forward_probe(model, device, linear_probe, batch, target_fn)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.update()
            pbar.set_description(
                f"Epoch {i + 1}/{args.n_epochs} {loss=:.4f} {test_accs.mean()=}"
            )
            if args.use_wandb and step >= args.warmup_steps:
                wandb.log({"train_loss": loss}, step=step)
            step += 1

        test_loss, test_accs = test_linear_probe(
            model, device, test_dataset, test_y, linear_probe, target_fn
        )

        if args.use_wandb:
            wandb.log({"eval_loss": test_loss, "eval_acc": test_accs.mean()}, step=step)

    if args.use_wandb:
        wandb.finish()

    return linear_probe


# %%
default_args = LinearProbeTrainingArgs(batch_size=128, n_test=200, use_wandb=False)
test_args = LinearProbeTrainingArgs(
    use_wandb=False, n_epochs=2, n_steps_per_epoch=10, lr=1e-2
)

runs = [
    # ("tem", theirs_empty_mine_target, default_args),
    # ("legal", legality_target, default_args),
    # ("ptem", prev_tem_target, default_args),
    ("cap", captures_target, default_args, ["flips"]),
    ("dir", flip_dir_target, default_args, ["flip_dirs"]),
    # ("pptem", lambda x, device: prev_tem_target(x, device, n_shift=2), default_args)
]

for name, fn, args, cols in runs:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = probe_dir / f"probe_{name}_{ts}.pt"
    model = load_model(device, "awonga/othello-gpt-7M")
    # args = test_args
    linear_probe = train_linear_probe(model, args, lambda x: fn(x, device), cols)
    t.save(
        linear_probe,
        save_path,
    )
    del linear_probe
    t.cuda.empty_cache()

# %%
