"""
this version is used for test presearch and time out. use presearch to help avoid timeout and add
a candidate according to level four presearch
"""

import numba
import numpy as np
import random
import time
import math

infinity = math.inf

BRANCH_FACTOR = 5

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

para1 = np.array([
    [4000, -18, 6, 4, 4, 6, -18, 4000],
    [-18, -9, 4, 3, 3, 4, -9, -18],
    [6, 4, 3, 2, 2, 3, 4, 6],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [6, 4, 3, 2, 2, 3, 4, 6],
    [-18, -9, 4, 3, 3, 4, -9, -18],
    [4000, -18, 6, 4, 4, 6, -18, 4000]
])

chessboard_size = 8

def evaluate(chessboard, color):
    idx1 = np.where(chessboard == color)  # 自己
    idx2 = np.where(chessboard == -color)  # 对手
    res = para1[idx1].sum() - para1[idx2].sum()  # 自己减对手的差
    return res


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.chessboard = np.zeros((chessboard_size, chessboard_size))
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your decision .
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard):
        self.candidate_list.clear()
        # start = time.time()
        self.chessboard = chessboard
        self.candidate_list = generateCandidate(chessboard, self.color)
        move = self.absearch()
        if move:
            self.candidate_list.append(move)
        # end = time.time()
        # print(end - start)

    def presearch(self):
        alpha = -infinity
        beta = infinity
        level = 4
        child_list = []
        for action in self.candidate_list:
            update_i, update_j = self.place(self.chessboard, action, self.color)
            _, utility = self.maximize(self.chessboard, alpha, beta, level - 1)
            self.undo_place(update_i, update_j, action, self.chessboard, self.color)
            child_list.append((utility, action))
        child_list.sort()
        return child_list

    def absearch(self):
        alpha = -infinity
        beta = infinity
        stage = np.count_nonzero(self.chessboard)
        start = time.time()
        child_list = self.presearch()
        end = time.time()
        presearch_time = end - start
        print(presearch_time)
        if len(child_list) != 0:
            self.candidate_list.append(child_list[0][1])
        if stage < 50:
            if presearch_time > 0.42:
                level = 5
            elif presearch_time < 0.08:
                level = 7
            else:
                level = 6
            move, _ = self.minimize(self.chessboard, alpha, beta, level)
        else:
            level = 8
            move, _ = self.minimize(self.chessboard, alpha, beta, level)
        return move

    def appro_absearch(self, child_list):
        alpha = -infinity
        beta = infinity
        minChild = None
        minUtility = infinity
        level = 6
        for i in range(0, BRANCH_FACTOR): #只搜前5个
            update_i, update_j = self.place(self.chessboard, child_list[i][1], self.color)
            _, utility = self.maximize(self.chessboard, alpha, beta, level - 1)
            self.undo_place(update_i, update_j, child_list[i][1], self.chessboard, self.color)
            if utility < minUtility:
                minChild = child_list[i][1]
                minUtility = utility
            if minUtility <= alpha:
                break
            beta = min(minUtility, beta)
        return minChild, minUtility

    def maximize(self, chessboard, alpha, beta, level):
        if level == 0:
            return None, evaluate(chessboard, self.color)
        maxChild = None
        maxUtility = -infinity
        children = generateCandidate(chessboard, -self.color)
        for child in children:
            update_i, update_j = self.place(chessboard, child, -self.color)
            _, utility = self.minimize(chessboard, alpha, beta, level - 1)
            self.undo_place(update_i, update_j, child, chessboard, -self.color)
            if utility > maxUtility:
                maxChild = child
                maxUtility = utility
            if maxUtility >= beta:
                break
            alpha = max(maxUtility, alpha)
        if len(children) == 0:
            _, utility = self.minimize(chessboard, alpha, beta, level - 1)
            return None, utility
        return maxChild, maxUtility

    def minimize(self, chessboard, alpha, beta, level):
        if level == 0:
            return None, evaluate(chessboard, self.color)
        minChild = None
        minUtility = infinity
        children = generateCandidate(chessboard, self.color)
        for child in children:
            update_i, update_j = self.place(chessboard, child, self.color)
            _, utility = self.maximize(chessboard, alpha, beta, level - 1)
            self.undo_place(update_i, update_j, child, chessboard, self.color)
            if utility < minUtility:
                minChild = child
                minUtility = utility
            if minUtility <= alpha:
                break
            beta = min(minUtility, beta)
        if len(children) == 0:
            _, utility = self.maximize(chessboard, alpha, beta, level - 1)
            return None, utility
        return minChild, minUtility

    def place(self, chessboard, move, color):
        i_arr = []
        j_arr = []
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for direction in directions:
            i_tempt = move[0]
            j_tempt = move[1]
            moved = False
            i_tempt_arr = []
            j_tempt_arr = []
            while 0 <= i_tempt + direction[0] < chessboard_size and \
                    0 <= j_tempt + direction[1] < chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == -color:
                i_tempt = i_tempt + direction[0]
                j_tempt = j_tempt + direction[1]
                moved = True
                i_tempt_arr.append(i_tempt)
                j_tempt_arr.append(j_tempt)
            if moved and 0 <= i_tempt + direction[0] < chessboard_size and \
                    0 <= j_tempt + direction[1] < chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == color:
                i_arr.extend(i_tempt_arr)
                j_arr.extend(j_tempt_arr)
        chessboard[(i_arr,j_arr)] = color
        chessboard[move] = color
        return i_arr, j_arr

    def undo_place(self, update_i, update_j, move, chessboard, color):
        chessboard[(update_i, update_j)] = -color
        chessboard[move[0]][move[1]] = COLOR_NONE


# generate candidate
@numba.njit(cache=True)
def generateCandidate(chessboard: np.ndarray, color: int):
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    candidate_list = []
    idx = np.where(chessboard == COLOR_NONE)
    idx = list(zip(idx[0], idx[1]))
    for position in idx:
        i = position[0]
        j = position[1]
        for direction in directions:
            i_tempt = i
            j_tempt = j
            moved = False
            while 0 <= i_tempt + direction[0] < chessboard_size and \
                    0 <= j_tempt + direction[1] < chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == -color:
                i_tempt = i_tempt + direction[0]
                j_tempt = j_tempt + direction[1]
                moved = True
            if moved and 0 <= i_tempt + direction[0] < chessboard_size and \
                    0 <= j_tempt + direction[1] < chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == color:
                candidate_list.append((i, j))
                break
    return candidate_list

