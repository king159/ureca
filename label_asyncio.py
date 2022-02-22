import csv

import chess.engine
import time

import asyncio
import chess
import chess.engine


def classify(score: float) -> int:
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


async def get_result(fen, semaphore) -> tuple:
    async with semaphore:
        transport, engine = await chess.engine.popen_uci("stockfish/stockfish12.exe")
        board = chess.Board(fen)
        info = await engine.analyse(board, chess.engine.Limit(depth=15), info=chess.engine.INFO_SCORE)
        await engine.quit()
        if type(info["score"].white()) is chess.engine.Mate:
            score = 100 if info["score"].white().mate() > 0 else -100
        elif type(info["score"].white()) is chess.engine.MateGivenType:
            score = 100 if info["score"].white().mate() == 0 else -100
        else:
            score = info["score"].white().score() / 100
        return score, fen


if __name__ == '__main__':
    print(time.ctime())
    s_time = time.time()
    with open("./data/fen.csv") as fen_file:
        reader = csv.reader(fen_file)
        fen_list = [row[1] for row in reader][270000:300000]

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    semaphore = asyncio.BoundedSemaphore(100)
    tasks = [get_result(fen, semaphore) for fen in fen_list]
    done, _ = loop.run_until_complete(asyncio.wait(tasks))
    r = [d.result() for d in done]
    loop.close()

    result = []
    for s, fen in r:
        result.append([fen, s, classify(s)])

    with open("./data/label.csv", 'a+', newline='\n') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(result)
    result_file.close()

    e_time = time.time()
    print(time.ctime())
    print((e_time - s_time) / 3600)
