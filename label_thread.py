import csv
import time
from concurrent.futures import ProcessPoolExecutor
import chess.engine


def get_class(score: float) -> int:
    if score >= 5:
        return 0
    elif 3 <= score < 5:
        return 1
    elif 1 <= score < 3:
        return 2
    elif -1 < score < 1:
        return 3
    elif -3 < score <= -1:
        return 4
    elif -5 < score <= -3:
        return 5
    elif score <= -5:
        return 6


def get_score(white) -> float:
    if type(white) is chess.engine.Mate:
        score = 100 if white.mate() > 0 else -100
    elif type(white) is chess.engine.MateGivenType:
        score = 100 if white.mate() == 0 else -100
    else:
        score = white.score() / 100
    return score


def get_result(fen: str) -> tuple:
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish13.exe")
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=15), info=chess.engine.INFO_SCORE)
    engine.quit()
    score = get_score(info["score"].white())
    return score, fen


def main(s: int) -> None:
    with open("./data/distinct.csv") as fen_file:
        reader = csv.reader(fen_file)
        fen_list = [row[1] for row in reader][int(s * 1e5):int((s + 1) * 1e5)]
    with ProcessPoolExecutor() as executor:
        r = executor.map(get_result, fen_list)
        result = []
    for score, fen in r:
        result.append([fen, score, get_class(score)])
    print("start writing")
    with open("./data/label_13.csv", 'a+', newline='\n') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(result)


if __name__ == '__main__':
    start = 53
    while start < 60:
        print(start)
        print(time.ctime())
        s_time = time.time()
        main(start)
        e_time = time.time()
        print((e_time - s_time) / 3600)
        start += 1
