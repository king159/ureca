import csv
import os

import chess.pgn
import chess.engine
from tqdm import trange


def pgn2fen(source_name: str, target_name: str) -> None:
    result = []
    # total 498737 games
    total = 40
    target_addr = os.path.join("data", target_name)
    source_addr = os.path.join("data", source_name)
    with open(source_addr, encoding="utf-8") as pgn_file:
        for _ in trange(total):
            game = chess.pgn.read_game(pgn_file)
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen = str(board.fen())
                result.append([fen])
    with open(target_addr, 'a+', newline='\n') as target_file:
        wr = csv.writer(target_file, dialect='excel')
        wr.writerows(result)


if __name__ == '__main__':
    pgn2fen("CB1.pgn", "CB1_label.csv")
