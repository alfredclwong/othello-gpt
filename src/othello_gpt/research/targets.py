import torch as t
import einops
from eindex import eindex
from jaxtyping import Union, Float
from typing import Tuple, List, Optional

from transformer_lens import HookedTransformer


def forward_probe(
    model: HookedTransformer,
    device,
    linear_probe: Float[t.Tensor, "d_model n_out d_probe layer"],
    batch,
    target_fn,
    return_loss=True,
    return_labels=False,
    scalar_loss=True,
) -> Tuple[
    Float[t.Tensor, "layer batch n_ctx n_out d_probe"],
    Union[t.Tensor, Optional[List[str]]],
]:
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
        "layer batch n_ctx d_model, d_model n_out d_probe layer -> layer batch n_ctx n_out d_probe",
    )
    log_probs = preds.log_softmax(-1)

    if not return_loss:
        if return_labels:
            return log_probs, labels
        return log_probs, None

    y = target_fn(batch)
    correct_log_probs = eindex(
        log_probs, y.int(), "layer batch n_ctx n_out [batch n_ctx n_out]"
    )
    mask = ~y.isnan()  # don't train on nan targets
    correct_log_probs = correct_log_probs.masked_fill(~mask, t.nan)
    loss = -correct_log_probs.nanmean() if scalar_loss else -correct_log_probs

    if return_labels:
        return log_probs, loss, labels
    else:
        return log_probs, loss


def next_move_target(batch, device) -> Float[t.Tensor, "batch pos n_out"]:
    coords = t.tensor(batch["coords"], device=device)[:, 1:]
    size = int(coords.max()) + 1
    next_move_board = t.zeros((*coords.shape[:-1], size, size), device=device)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            y, x = coords[i, j]
            next_move_board[i, j, y, x] = 1
    return next_move_board.flatten(2)


def empty_target(batch, device):
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    return (boards == 0).flatten(2)


def t_npt_target(batch, device, pad_nan=True):
    # Theirs | not previously theirs
    # Parities are a bit of a mess. Let's get this one right:
    # Black plays firsts in Othello, I represent B/E/W as +1/0/-1
    # In plot_game, I use gray_r to get the correct colours which is confusing
    # For the T/E/M transform, pos 0 of a game looks at the board state after the
    # first move. This means that T = B, M = W. We want T = +1 and M = -1, so we
    # flip the sign on the odd indices. Good, that's what I've been doing.
    # (I've also previously had T = 0, E = 1, M = 2 for the classifier....!?)
    # Note that PT/PM is the opposite. confusedjackiechan.jpg

    boards = t.tensor(batch["boards"], device=device)[:, :-1]

    # Prepend with initial board (BW)
    n_batch = boards.shape[0]
    size = boards.shape[-1]
    initial_board = t.zeros((n_batch, 1, size, size), device=device)
    i = size // 2 - 1
    initial_board[:, -1, [i, i + 1], [i, i + 1]] = 1  # B
    initial_board[:, -1, [i, i + 1], [i + 1, i]] = -1  # W
    boards = t.cat([initial_board, boards], dim=1)

    # Shift BW to TM (we flip even indices now because we prepended initial board)
    boards[:, ::2] *= -1

    # Get target
    theirs = boards[:, 1:] == 1
    nptheirs = boards[:, :-1] != -1
    target = t.where(nptheirs, theirs, t.nan)

    if pad_nan:
        target[:, 0] = t.nan

    return target.flatten(2)

def prev_empty_target(batch, device, n_shift=1):
    e = empty_target(batch, device)
    n_batch = e.shape[0]
    n_out = e.shape[-1]
    e0 = t.full((n_batch, n_shift, n_out), t.nan, device=device)
    return t.cat([e0, e[:, :-n_shift]], dim=1)


def tm_target(batch, device):
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    boards[:, 1::2] *= -1
    boards = t.where(boards == 0, t.nan, boards == 1)
    return boards.flatten(2)


def ptm_target(batch, device, n_shift=1):
    tm = tm_target(batch, device)
    n_batch = tm.shape[0]
    n_out = tm.shape[-1]
    initial_board = t.full((n_batch, n_shift, n_out), t.nan, device=device)
    return t.cat([initial_board, 1-tm[:, :-n_shift]], dim=1)


def bw_target(batch, device):
    # TODO check sign
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    boards = t.where(boards == 0, t.nan, boards == 1)
    return boards.flatten(2)


def theirs_empty_mine_target(batch, device) -> Float[t.Tensor, "batch pos n_out"]:
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    boards[:, 1::2] *= -1
    boards += 1
    return boards.flatten(2)


def prev_tem_target(batch, device, n_shift=1, pad_nan=True) -> Float[t.Tensor, "batch pos n_out"]:
    tem = theirs_empty_mine_target(batch, device)

    n_batch = tem.shape[0]
    n_out = tem.shape[-1]
    size = int((n_out + 4) ** 0.5)

    if pad_nan:
        initial_board = t.full((n_batch, n_shift, n_out), t.nan, device=device)
    else:
        initial_board = t.zeros((n_batch, n_shift, size, size), device=device)
        i = size // 2 - 1
        initial_board[:, -1, [i, i + 1], [i, i + 1]] = -1  # TODO check parity
        initial_board[:, -1, [i, i + 1], [i + 1, i]] = 1
        initial_board = initial_board.flatten(2) + 1

    return t.cat([initial_board, tem[:, :-n_shift]], dim=1)


def legality_target(batch, device):
    return t.tensor(batch["legalities"], device=device)[:, 1:].flatten(2)


def l_if_e_target(batch, device):
    e = empty_target(batch, device)
    l = legality_target(batch, device)
    return t.where(e.bool(), l, t.nan)


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
    return t.tensor(batch["flips"], device=device)[:, :-1].int().flatten(2)


def c_if_ne_target(batch, device):
    e = empty_target(batch, device)
    c = captures_target(batch, device)
    return t.where(~e.bool(), c, t.nan)


def c_if_t_target(batch, device):
    # TODO c_if_ray_target
    # explanation: i think the model embeds capture vectors into every square
    # that a move `could` capture (or a reasonable average prior) and then
    # corrects this in subsequent layers, so we can use a stricter condition
    # than non-empty, theirs or prev mine to further isolate the probe
    met = theirs_empty_mine_target(batch, device)
    c = captures_target(batch, device)
    return t.where(met == 2, c, t.nan)


def c_if_pm_target(batch, device):
    ptem = prev_tem_target(batch, device, pad_nan=False)
    c = captures_target(batch, device)
    return t.where(ptem == 2, c, t.nan)


# def tem_captures_target(batch, device):
#     # Flip sign of captures to represent their vs my captures
#     flips = captures_target(batch, device)
#     flips[:, 1::2] *= -1
#     flips += 1
#     return flips


def flip_dir_target(batch, device) -> Float[t.Tensor, "batch pos n_out"]:
    return t.tensor(batch["flip_dirs"], device=device)[:, :-1].int().flatten(2)


# def original_colour_target(batch, device):
#     # # Hypothesis: moves -> H0 -> (original colour, flips) -> M0 -> (originally mine & not flipped = mine, originally mine & flipped = theirs, empty = empty, etc.) -> H1 -> (?) -> M1 -> (legal)
#     return t.tensor(batch["originals"], device=device)[:, :-1].int() + 1
