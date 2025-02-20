import torch as t
import einops
from eindex import eindex
from jaxtyping import Union
from typing import Tuple, List

def theirs_empty_mine_target(batch, device):
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    boards[:, 1::2] *= -1
    boards += 1
    return boards


def prev_tem_target(batch, device):
    tem = theirs_empty_mine_target(batch, device)

    n_batch = tem.shape[0]
    size = tem.shape[-1]

    initial_board = t.zeros((n_batch, 1, size, size), device=device)
    # i = size // 2 - 1
    # initial_board[..., [i, i + 1], [i, i + 1]] = -1
    # initial_board[..., [i, i + 1], [i + 1, i]] = 1

    return t.cat([initial_board + 1, tem[:, :-1]], dim=1).int()


def legality_target(batch, device):
    return t.tensor(batch["legalities"], device=device)[:, 1:].int()


# def flip_parity_target(batch, device):
#     # At each game position, a non-empty tile is either the same colour as when it was first played (0)
#     # or it has been flipped to the other colour (1). I think that this is a necessary state for the
#     # model to track in order to have an accurate board state representation.
#     flips = t.tensor(batch["flips"], device=device)[:, :-1].int()
#     return flips.cumsum(dim=1) % 2


# def mine_flip_target(batch, device):
#     # TODO (why) is initial accuracy on last layer significantly higher?
#     # Hypothesis: model tracks tiles that got flipped, grouped by mine/theirs
#     flip_parity = flip_parity_target(batch, device)
#     tem = theirs_empty_mine_target(batch, device)
#     return t.logical_and(flip_parity, (tem == 2)).int()


# def omine_flip_target(batch, device):
#     # TODO (why) is initial accuracy on last layer significantly higher?
#     # Hypothesis: model tracks tiles that got flipped, grouped by mine/theirs
#     flip_parity = flip_parity_target(batch, device)
#     otem = original_colour_target(batch, device)
#     return t.logical_and(flip_parity, (otem == 2)).int()


def captures_target(batch, device):
    # Hypothesis: each token tracks the tiles that it captured when the move was played
    # After H0, we have [my moves; their moves; my moves flipped; their moves flipped]
    # This gives us the
    return t.tensor(batch["flips"], device=device)[:, :-1].int()


def tem_captures_target(batch, device):
    # Flip sign of captures to represent their vs my captures
    flips = captures_target(batch, device)
    flips[:, 1::2] *= -1
    flips += 1
    return flips


# def original_colour_target(batch, device):
#     # # Hypothesis: moves -> H0 -> (original colour, flips) -> M0 -> (originally mine & not flipped = mine, originally mine & flipped = theirs, empty = empty, etc.) -> H1 -> (?) -> M1 -> (legal)
#     return t.tensor(batch["originals"], device=device)[:, :-1].int() + 1


def forward_probe(
    model, device, linear_probe, batch, target_fn, return_loss=True, return_labels=False
) -> Tuple[t.Tensor, Union[t.Tensor, List]]:
    # input_ids = pad_batch(batch["input_ids"], max_len=model.cfg.n_ctx + 1).to(device)
    input_ids = t.tensor(batch["input_ids"], device=device)
    _, cache = model.run_with_cache(
        input_ids[:, :-1],
        names_filter=lambda name: "hook_resid_" in name
        or "ln_final.hook_scale" in name,
    )
    X, labels = cache.accumulated_resid(
        apply_ln=True, incl_mid=True, return_labels=True
    )

    preds = einops.einsum(
        X,
        linear_probe,
        "layer batch n_ctx d_model, d_model row col d_probe layer -> layer batch n_ctx row col d_probe",
    )
    log_probs = preds.log_softmax(-1)

    if not return_loss:
        if return_labels:
            return log_probs, labels
        return log_probs, None

    y = target_fn(batch)
    correct_log_probs = eindex(
        log_probs, y, "layer batch n_ctx rows cols [batch n_ctx rows cols]"
    )
    loss = -correct_log_probs.mean()

    return log_probs, loss
