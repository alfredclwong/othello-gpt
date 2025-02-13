import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, List

import huggingface_hub as hf
import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from othello_gpt.data.vis import move_id_to_coord, move_id_to_text
from othello_gpt.othello import OthelloState, get_legal_move_ids, is_terminal, make_move
from othello_gpt.util import tokenize


def generate_game(size: int, no_pass: bool = True) -> Dict[str, List]:
    legalities = []
    moves = []
    coords = []
    squares = []
    boards = []
    flips = []

    state = OthelloState(size)

    while not is_terminal(state):
        legal_move_ids = get_legal_move_ids(state, no_pass=no_pass)

        if not legal_move_ids:
            return generate_game(size, no_pass=no_pass)

        legalities.append(np.zeros((size, size), dtype=bool))
        for move_id in legal_move_ids:
            if move_id != size * size:
                legalities[-1][*divmod(move_id, size)] = 1

        move_id = np.random.choice(legal_move_ids)
        moves.append(move_id)
        coords.append(move_id_to_coord(move_id, size))
        squares.append(move_id_to_text(move_id, size))

        state, _flips = make_move(state, move_id, validate=False)
        boards.append(state.board)
        flips.append(np.zeros((size, size), dtype=bool))
        for y, x in _flips:
            flips[-1][y, x] = 1

    return {
        "legalities": legalities,  # these are the legal squares for the current move (i.e. previous board!) TODO shift and refactor
        "moves": moves,
        "coords": coords,
        "squares": squares,
        "boards": boards,
        "flips": flips,
    }


def generate_batch(args) -> Path:
    # batch_id: int, size: int, no_pass: bool, batch_size: int, tmp_dir: Path
    batch_id, size, no_pass, batch_size, tmp_dir = args
    pbar = tqdm(range(batch_size), desc=f"{batch_id=}", position=batch_id + 1)
    games = [generate_game(size, no_pass) for _ in pbar]
    games = {k: [game[k] for game in games] for k in games[0].keys()}
    dataset = Dataset.from_dict(games)
    tmp_file = tmp_dir / f"batch_{batch_id}.arrow"
    dataset.save_to_disk(tmp_file)
    return tmp_file


def generate_dataset(
    tmp_dir: Path,
    n_games: int,
    size: int,
    batch_size: int = 10000,
    no_pass: bool = True,
) -> Dataset:
    n_batch = (n_games - 1) // batch_size + 1
    tmp_files: List[Path] = process_map(
        generate_batch,
        [(i, size, no_pass, batch_size, tmp_dir) for i in range(n_batch)],
        max_workers=mp.cpu_count(),
    )

    datasets = [
        Dataset.load_from_disk(tmp_file) for tmp_file in tqdm(tmp_files, desc="Loading")
    ]
    final_dataset = concatenate_datasets(datasets)

    return final_dataset


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path().cwd()
    hf.login((root_dir / "secret.txt").read_text())

    n_games = 2000000
    size = 6

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = generate_dataset(Path(tmp_dir), n_games, size)
        dataset_dict = dataset.train_test_split(test_size=0.1)
        dataset_dict["test"] = dataset_dict["test"].map(
            lambda x: tokenize(x["moves"], size),
            batched=True,
        )
        dataset_dict["train"] = dataset_dict["train"].map(
            lambda x: tokenize(x["moves"], size),
            batched=True,
        )
        dataset_dict.push_to_hub("awonga/othello-gpt")
