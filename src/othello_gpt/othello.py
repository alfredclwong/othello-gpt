import numpy as np
from typing import List, Optional, Tuple

players = [1, -1]
opponent = {1: -1, -1: 1}


def get_max_moves(size: int) -> int:
    return (size * size - 4) * 2  # Upper bound assuming one player only passes


class OthelloState:  # TODO immutable dataclass?
    turn: int
    size: int
    board: np.ndarray
    history: np.ndarray

    def __init__(
        self,
        size: Optional[int] = None,
        board: Optional[np.ndarray] = None,
        history: Optional[np.ndarray] = None,
        turn: Optional[int] = None,
    ):
        if size is not None and board is None and history is None and turn is None:
            self.size = size

            self.board = np.zeros((size, size), dtype=int)
            self.board[size // 2 - 1, size // 2 - 1] = players[0]
            self.board[size // 2, size // 2] = players[0]
            self.board[size // 2 - 1, size // 2] = players[1]
            self.board[size // 2, size // 2 - 1] = players[1]

            max_moves = get_max_moves(self.size)
            self.history = -np.ones(max_moves, dtype=int)

            self.turn = players[0]
        elif (
            size is None
            and board is not None
            and history is not None
            and turn is not None
        ):
            self.size = board.shape[0]
            self.board = board
            self.history = history
            self.turn = turn
        else:
            raise ValueError("Invalid arguments")

    def __str__(self):
        id_history = self.history[self.history != -1]
        coord_history = [(divmod(int(i), self.size)) for i in id_history]
        square_history = [
            "PASS" if y == self.size else f"{chr(ord('a') + x)}{y + 1}"
            for y, x in coord_history
        ]
        histories = {"id": id_history, "coord": coord_history, "square": square_history}
        histories_str = "\n".join(f"{k}_history = {v}" for k, v in histories.items())
        return f"Turn = {self.turn}\n{histories_str}\n{self.board}"


def get_flips(state: OthelloState, move: int) -> List[Tuple[int, int]]:
    turn = state.turn
    size = state.size
    flips = []
    directions = [
        (-1, -1),  # NW
        (-1, 0),  # N
        (-1, 1),  # NE
        (0, -1),  # W
        (0, 1),  # E
        (1, -1),  # SW
        (1, 0),  # S
        (1, 1),  # SE
    ]
    for direction in directions:
        flip = []
        y, x = divmod(move, size)
        dy, dx = direction
        y += dy
        x += dx
        while 0 <= y < size and 0 <= x < size and state.board[y, x] == opponent[turn]:
            flip.append((y, x))
            y += dy
            x += dx
        if 0 <= y < size and 0 <= x < size and state.board[y, x] == turn:
            flips.extend(flip)
    return flips


def make_move(
    state: OthelloState, move: int, validate: bool = True
) -> Tuple[OthelloState, List[Tuple[int, int]]]:
    board = state.board.copy()

    history = state.history.copy()
    history[(history < 0).argmax()] = move

    turn = opponent[state.turn]

    if move == state.size * state.size:
        return OthelloState(board=board, history=history, turn=turn)

    flips = get_flips(state, move)
    if validate and (
        len(flips) == 0 or state.board[move // state.size, move % state.size] != 0
    ):
        raise ValueError(f"Invalid move: {move}")

    board[move // state.size, move % state.size] = state.turn
    for y, x in flips:
        board[y, x] = state.turn

    return OthelloState(board=board, history=history, turn=turn), flips


def get_legal_move_ids(state: OthelloState, no_pass: bool = True) -> List[int]:
    legal_moves = []
    for move in range(state.size * state.size):
        if (
            state.board[move // state.size, move % state.size] == 0
            and len(get_flips(state, move)) > 0
        ):
            legal_moves.append(move)
    if not legal_moves and not no_pass:
        legal_moves.append(state.size * state.size)
    return legal_moves


def is_terminal(state: OthelloState) -> bool:
    pass_id = state.size * state.size
    n_moves = (state.history >= 0).sum()
    is_board_full = (state.board != 0).all()
    two_consecutive_passes = (
        (n_moves >= 2)
        and (state.history[n_moves - 1] == pass_id)
        and (state.history[n_moves - 2] == pass_id)
    )
    return is_board_full or two_consecutive_passes
