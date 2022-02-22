import re
import functools
import operator
from typing import List

import numpy as np
from chess import Board
from chess import Piece
import chess
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import csv
from tqdm import trange, tqdm
from concurrent.futures import ProcessPoolExecutor

PIECE_SCORE = {6: 1350, 5: 975, 4: 500, 3: 325, 2: 325, 1: 100}


def least_attack(board, color):
    k = board.pieces(chess.KING, color)
    q = board.pieces(chess.QUEEN, color)
    r = board.pieces(chess.ROOK, color)
    b = board.pieces(chess.BISHOP, color)
    n = board.pieces(chess.KNIGHT, color)
    p = board.pieces(chess.PAWN, color)
    attack_board = Board("8/8/8/8/8/8/8/8")
    while k:
        K = k.pop()
        for s in list(board.attacks(K)):
            attack_board.set_piece_at(s, Piece(chess.KING, color))
    while q:
        Q = q.pop()
        for s in list(board.attacks(Q)):
            attack_board.set_piece_at(s, Piece(chess.QUEEN, color))
    while r:
        R = r.pop()
        for s in list(board.attacks(R)):
            attack_board.set_piece_at(s, Piece(chess.ROOK, color))
    while b:
        B = b.pop()
        for s in list(board.attacks(B)):
            attack_board.set_piece_at(s, Piece(chess.BISHOP, color))
    while n:
        N = n.pop()
        for s in list(board.attacks(N)):
            attack_board.set_piece_at(s, Piece(chess.KNIGHT, color))
    while p:
        P = p.pop()
        for s in list(board.attacks(P)):
            attack_board.set_piece_at(s, Piece(chess.PAWN, color))
    control_list = [0] * 64
    for i in range(64):
        if attack_board.piece_at(i) is not None:
            control_list[i] = (2025 - PIECE_SCORE[attack_board.piece_at(i).piece_type]) / 2700
    return attack_board, control_list


class AttackMap:
    W_attack = Board("8/8/8/8/8/8/8/8")
    B_attack = Board("8/8/8/8/8/8/8/8")
    W_control = [0.0] * 64
    B_control = [0.0] * 64

    def __init__(self, board: Board):
        self.W_attack, self.W_control = least_attack(board, chess.WHITE)
        self.B_attack, self.B_control = least_attack(board, chess.BLACK)


def getx(b: int) -> float:
    return b % 8


def gety(b: int) -> float:
    return b // 8


def getx_n(b: int) -> float:
    return (b % 8) * 0.1429


def gety_n(b: int) -> float:
    return (b // 8) * 0.1429


def get_movability_parallel(pos: int, board: Board) -> list:
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    go = list(board.attacks(pos))
    for p in go:
        if getx(p) == getx(pos) and gety(p) > gety(pos):
            x1 += 1
        elif getx(p) == getx(pos) and gety(p) < gety(pos):
            x2 += 1
        elif gety(p) == gety(pos) and getx(p) < getx(pos):
            y1 += 1
        elif gety(p) == gety(pos) and getx(p) > getx(pos):
            y2 += 1
    return [x1 / 7, x2 / 7, y1 / 7, y2 / 7]


def get_movability_oblique(pos: int, board: Board) -> list:
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    go = list(board.attacks(pos))
    for p in go:
        if (getx(p) - getx(pos)) == (gety(p) - gety(pos)) and (getx(p) - getx(pos)) > 0:
            x1 += 1
        elif (getx(p) - getx(pos)) == (gety(p) - gety(pos)) and (getx(p) - getx(pos)) < 0:
            x2 += 1
        elif getx(p) - getx(pos) == gety(pos) - gety(p) and (getx(pos) - getx(p)) > 0:
            y1 += 1
        elif getx(p) - getx(pos) == gety(pos) - gety(p) and (getx(pos) - getx(p)) < 0:
            y2 += 1
    return [x1 / 7, x2 / 7, y1 / 7, y2 / 7]


def safe_count(pos: int, board: Board, a_map: AttackMap) -> float:
    p = board.piece_at(pos)
    count = 0
    if p.color == chess.WHITE:
        attack = a_map.B_attack
    else:
        attack = a_map.W_attack
    for i in list(board.attacks(pos)):
        opp = attack.piece_at(i)
        if opp is None:
            count += 1
        elif PIECE_SCORE[int(opp.piece_type)] >= PIECE_SCORE[int(p.piece_type)] and board.is_attacked_by(p.color,
                                                                                                         pos) > 1:
            count += 1
    return count / 16


PIECES = {"K": 1, "Q": 2, "R": 3, "B": 4, "N": 5, "P": 6, "k": 7, "q": 8, "r": 9, "b": 10, "n": 11, "p": 12}


def fen2vec_manual(board: chess.Board) -> torch.Tensor:
    # material
    fen = board.fen()
    ma = re.split(r' ', fen)[0]
    ma_vector = [0.0] * 10
    for p in ma:
        if p == 'Q':
            ma_vector[0] += 1
        if p == 'R':
            ma_vector[1] += 1
        if p == 'B':
            ma_vector[2] += 1
        if p == 'N':
            ma_vector[3] += 1
        if p == 'P':
            ma_vector[4] += 1
        if p == 'q':
            ma_vector[5] += 1
        if p == 'r':
            ma_vector[6] += 1
        if p == 'b':
            ma_vector[7] += 1
        if p == 'n':
            ma_vector[8] += 1
        if p == 'p':
            ma_vector[9] += 1
    for i in [1, 2, 3, 6, 7, 8]:
        ma_vector[i] /= 2
    for j in [4, 9]:
        ma_vector[j] /= 8

    # side
    side_vector = [1.0 if ma[1] == 'w' else 0.0]

    # king
    king_vector = [0.0] * 8
    WK_pos = list(board.pieces(chess.KING, chess.WHITE))[0]
    BK_pos = list(board.pieces(chess.KING, chess.BLACK))[0]
    king_vector[0] = getx_n(WK_pos)
    king_vector[1] = gety_n(WK_pos)
    king_vector[2] = int(board.has_kingside_castling_rights(chess.WHITE))
    king_vector[3] = int(board.has_queenside_castling_rights(chess.WHITE))
    king_vector[4] = getx_n(BK_pos)
    king_vector[5] = gety_n(BK_pos)
    king_vector[6] = int(board.has_kingside_castling_rights(chess.BLACK))
    king_vector[7] = int(board.has_queenside_castling_rights(chess.BLACK))

    a_map = AttackMap(board)

    # pawn
    pawn_vector = [0.0] * 80
    WP_list = list(board.pieces(chess.PAWN, chess.WHITE))
    for i, pos in enumerate(WP_list):
        pawn_vector[i * 5] = 1.0
        pawn_vector[i * 5 + 1] = getx_n(pos)
        pawn_vector[i * 5 + 2] = gety_n(pos)
        pawn_vector[i * 5 + 3] = a_map.W_control[pos]
        pawn_vector[i * 5 + 4] = a_map.B_control[pos]
    BP_list = list(board.pieces(chess.PAWN, chess.BLACK))
    for j, pos in enumerate(BP_list):
        pos = BP_list[j]
        pawn_vector[j * 5 + 40] = 1.0
        pawn_vector[j * 5 + 41] = getx_n(pos)
        pawn_vector[j * 5 + 42] = gety_n(pos)
        pawn_vector[j * 5 + 43] = a_map.B_control[pos]
        pawn_vector[j * 5 + 44] = a_map.W_control[pos]

    # bishop
    bishop_vector = [0.0] * 40
    WB_list = list(board.pieces(chess.BISHOP, chess.WHITE))
    for i, pos in enumerate(WB_list):
        if i > 1:
            break
        bishop_vector[i * 10] = 1
        bishop_vector[i * 10 + 1] = getx_n(pos)
        bishop_vector[i * 10 + 2] = gety_n(pos)
        bishop_vector[i * 10 + 3] = a_map.W_control[pos]
        bishop_vector[i * 10 + 4] = a_map.B_control[pos]
        bishop_vector[i * 10 + 5] = safe_count(pos, board, a_map)
        m = get_movability_oblique(pos, board)
        bishop_vector[i * 10 + 6] = m[0]
        bishop_vector[i * 10 + 7] = m[1]
        bishop_vector[i * 10 + 8] = m[2]
        bishop_vector[i * 10 + 9] = m[3]
    BB_list = list(board.pieces(chess.BISHOP, chess.BLACK))
    for i, pos in enumerate(BB_list):
        if i > 1:
            break
        pos = BB_list[i]
        bishop_vector[i * 10 + 20] = 1
        bishop_vector[i * 10 + 21] = getx_n(pos)
        bishop_vector[i * 10 + 22] = gety_n(pos)
        bishop_vector[i * 10 + 23] = a_map.B_control[pos]
        bishop_vector[i * 10 + 24] = a_map.W_control[pos]
        bishop_vector[i * 10 + 25] = safe_count(pos, board, a_map)
        m = get_movability_oblique(pos, board)
        bishop_vector[i * 10 + 26] = m[0]
        bishop_vector[i * 10 + 27] = m[1]
        bishop_vector[i * 10 + 28] = m[2]
        bishop_vector[i * 10 + 29] = m[3]

    # queen
    queen_vector = [0.0] * 28
    WQ_pos = list(board.pieces(chess.QUEEN, chess.WHITE))
    if len(WQ_pos) > 0:
        pos = WQ_pos[0]
        queen_vector[0] = 1
        queen_vector[1] = getx_n(pos)
        queen_vector[2] = gety_n(pos)
        queen_vector[3] = a_map.W_control[pos]
        queen_vector[4] = a_map.B_control[pos]
        queen_vector[5] = safe_count(pos, board, a_map)
        m1 = get_movability_oblique(pos, board)
        m2 = get_movability_parallel(pos, board)
        queen_vector[6] = m1[0]
        queen_vector[7] = m1[1]
        queen_vector[8] = m1[2]
        queen_vector[9] = m1[3]
        queen_vector[10] = m2[0]
        queen_vector[11] = m2[1]
        queen_vector[12] = m2[2]
        queen_vector[13] = m2[3]
    BQ_pos = list(board.pieces(chess.QUEEN, chess.BLACK))
    if len(BQ_pos) > 0:
        pos = BQ_pos[0]
        queen_vector[14] = 1
        queen_vector[15] = getx_n(pos)
        queen_vector[16] = gety_n(pos)
        queen_vector[17] = a_map.B_control[pos]
        queen_vector[18] = a_map.W_control[pos]
        queen_vector[19] = safe_count(pos, board, a_map)
        m1 = get_movability_oblique(pos, board)
        m2 = get_movability_parallel(pos, board)
        queen_vector[20] = m1[0]
        queen_vector[21] = m1[1]
        queen_vector[22] = m1[2]
        queen_vector[23] = m1[3]
        queen_vector[24] = m2[0]
        queen_vector[25] = m2[1]
        queen_vector[26] = m2[2]
        queen_vector[27] = m2[3]

    # knight
    knight_vector = [0.0] * 24
    WN_list = list(board.pieces(chess.KNIGHT, chess.WHITE))
    for i, pos in enumerate(WN_list):
        if i > 1:
            break
        knight_vector[i * 6] = 1
        knight_vector[i * 6 + 1] = getx_n(pos)
        knight_vector[i * 6 + 2] = gety_n(pos)
        knight_vector[i * 6 + 3] = a_map.W_control[pos]
        knight_vector[i * 6 + 4] = a_map.B_control[pos]
        knight_vector[i * 6 + 5] = safe_count(pos, board, a_map)
    BN_list = list(board.pieces(chess.KNIGHT, chess.BLACK))
    for i, pos in enumerate(BN_list):
        if i > 1:
            break
        knight_vector[i * 6 + 12] = 1
        knight_vector[i * 6 + 13] = getx_n(pos)
        knight_vector[i * 6 + 14] = gety_n(pos)
        knight_vector[i * 6 + 15] = a_map.B_control[pos]
        knight_vector[i * 6 + 16] = a_map.W_control[pos]
        knight_vector[i * 6 + 17] = safe_count(pos, board, a_map)

    # rook
    rook_vector = [0.0] * 44
    WR_list = list(board.pieces(chess.ROOK, chess.WHITE))
    for i, pos in enumerate(WR_list):
        if i > 1:
            break
        rook_vector[i * 10] = 1
        rook_vector[i * 10 + 1] = getx_n(pos)
        rook_vector[i * 10 + 2] = gety_n(pos)
        rook_vector[i * 10 + 3] = a_map.B_control[pos]
        rook_vector[i * 10 + 4] = a_map.W_control[pos]
        rook_vector[i * 10 + 5] = safe_count(pos, board, a_map)
        m = get_movability_parallel(pos, board)
        rook_vector[i * 10 + 6] = m[0]
        rook_vector[i * 10 + 7] = m[1]
        rook_vector[i * 10 + 8] = m[2]
        rook_vector[i * 10 + 9] = m[3]
    rook_vector[20] = int(board.has_kingside_castling_rights(chess.WHITE))
    rook_vector[21] = int(board.has_queenside_castling_rights(chess.WHITE))
    BR_list = list(board.pieces(chess.ROOK, chess.BLACK))
    for i, pos in enumerate(BR_list):
        if i > 1:
            break
        rook_vector[i * 10 + 22] = 1
        rook_vector[i * 10 + 23] = getx_n(pos)
        rook_vector[i * 10 + 24] = gety_n(pos)
        rook_vector[i * 10 + 25] = a_map.W_control[pos]
        rook_vector[i * 10 + 26] = a_map.B_control[pos]
        rook_vector[i * 10 + 27] = safe_count(pos, board, a_map)
        m = get_movability_parallel(pos, board)
        rook_vector[i * 10 + 28] = m[0]
        rook_vector[i * 10 + 29] = m[1]
        rook_vector[i * 10 + 30] = m[2]
        rook_vector[i * 10 + 31] = m[3]
    rook_vector[42] = int(board.has_kingside_castling_rights(chess.BLACK))
    rook_vector[43] = int(board.has_queenside_castling_rights(chess.BLACK))

    # square
    square_vector = a_map.W_control + a_map.B_control
    print(len(square_vector))
    result = ma_vector + side_vector + king_vector + pawn_vector + queen_vector + rook_vector + bishop_vector + \
             knight_vector + square_vector
    return torch.tensor(result)


def fen2vec_cnn12(board: chess.Board) -> torch.Tensor:
    """12*8*8"""
    if board.turn == chess.WHITE:
        vector = np.full((768, 1), 0.0)
    else:
        vector = np.full((768, 1), 1 / 14)
    for pos, p in board.piece_map().items():
        i = PIECES[str(p)] + 1
        vector[(i - 2) * 64 + pos] = i / 14
    vector = torch.from_numpy(vector)
    return vector.reshape(12, 8, 8).type('torch.FloatTensor')


def fen2vec_cnn2(board: chess.Board) -> torch.Tensor:
    """2*8*8"""
    if board.turn == chess.WHITE:
        vector = np.full((128, 1), 0.0)
    else:
        vector = np.full((128, 1), 1 / 14)
    for pos, p in board.piece_map().items():
        i = PIECES[str(p)] + 1
        if i < 8:
            vector[pos] = i / 14
        else:
            vector[pos + 64] = i / 14
    vector = torch.from_numpy(vector)
    return vector.reshape(2, 8, 8).type('torch.FloatTensor')


def fen2vec_cnn1(board: chess.Board) -> torch.Tensor:
    """1*8*8"""
    if board.turn == chess.WHITE:
        vector = np.full((64, 1), 0.0)
    else:
        vector = np.full((64, 1), 1 / 14)
    for pos, p in board.piece_map().items():
        i = PIECES[str(p)] + 1
        vector[pos] = i / 14
    vector = torch.from_numpy(vector)
    return vector.reshape(1, 8, 8).type('torch.FloatTensor')


def fen2vec_auto(board: chess.Board) -> torch.Tensor:
    vector = np.zeros((773, 1)).astype(np.float32)
    for pos, p in board.piece_map().items():
        i = PIECES[str(p)] - 1
        vector[64 * i + pos] = 1
    if board.turn == chess.WHITE:
        vector[768] = 1
    else:
        vector[768] = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        vector[769] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        vector[770] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        vector[771] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        vector[772] = 1
    vector = torch.from_numpy(vector)
    return vector.reshape(773)


def plot_cm(net, dataloader) -> None:
    y_pred = []
    y_true = []
    device = torch.device('cuda')
    net.eval()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
        y_pred.append(preds.tolist())
        y_true.append(labels.tolist())
    y_pred = functools.reduce(operator.iconcat, y_pred, [])
    y_true = functools.reduce(operator.iconcat, y_true, [])
    cm = confusion_matrix(y_true, y_pred)
    classes = ('ww', 'wa', 'wb', 'eq', 'bb', 'ba', 'bw')
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    # plt.title(t)
    plt.show()


def plot_sta(sta: dict, interval: int = None, title: str = None) -> None:
    """plot the statistics from the dictionary"""
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    epochs = len(sta['train']['epoch_acc'])
    max_d_acc = max(sta['dev']['epoch_acc'])
    max_d_acc_index = sta['dev']['epoch_acc'].index(max_d_acc) + 1

    # plot acc
    ax = plt.subplot(121)
    t_acc = sta['train']['epoch_acc']
    d_acc = sta['dev']['epoch_acc']
    tline, = plt.plot(np.append(np.roll(t_acc, 1), t_acc[epochs - 1]), color='g')
    dline, = plt.plot(np.append(np.roll(d_acc, 1), d_acc[epochs - 1]), linestyle=":", color='r')
    plt.grid(color="k", linestyle=":")
    plt.legend((tline, dline), ('train', 'dev'))
    plt.ylabel('acc')
    plt.xlabel('iterations')
    ax.set_xlim(1, epochs)
    if interval is not None:
        dim = np.arange(1, epochs + 1, interval)
        plt.xticks(dim)
    plt.scatter(max_d_acc_index, max_d_acc, s=40, color='black')

    # plot loss
    t_loss = sta['train']['epoch_loss']
    d_loss = sta['dev']['epoch_loss']
    ax = plt.subplot(122)
    tlline, = plt.plot(np.append(np.roll(t_loss, 1), t_loss[epochs - 1]), color='g')
    dlline, = plt.plot(np.append(np.roll(d_loss, 1), d_loss[epochs - 1]), linestyle=":", color='r')
    plt.grid(color="k", linestyle=":")
    plt.legend((tlline, dlline), ('train', 'dev'))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    ax.set_xlim(1, epochs)
    if interval is not None:
        dim = np.arange(1, epochs + 1, interval)
        plt.xticks(dim)

    plt.show()

    print("max train acc: " + str(max(t_acc)))
    print("max dev acc: " + str(max_d_acc) + " at epoch " + str(max_d_acc_index))
    print("corresponding train acc: " + str(t_acc[max_d_acc_index - 1]))


def topk_acc(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> int:
    with torch.no_grad():
        _, y_pred = output.topk(k=topk, dim=1)
        y_pred = y_pred.t()
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)
        ind_which_topk_matched_truth = correct[:topk]
        flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1)
        tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True).item()
    return int(tot_correct_topk)
