"""
this version is used for test presearch and time out. use presearch to help avoid timeout and add
a candidate according to level four presearch.
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

chessboard_size = 8


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size=8, color=COLOR_NONE, time_out=5.0):
        self.chessboard_size = chessboard_size
        self.chessboard = np.zeros((chessboard_size, chessboard_size))
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.black_weight_vector = [1, 2, 6.9802668786964235, 6.955834433152874, 5.955936082072979, 9.176010335696908, 12.358021682016863, 1.0488993643373625, -1.784330415039845, 57.80395980897628]
        self.white_weight_vector = [1, 3.469257198353225, 9.084906165858914, 5.476656155454725, 5.955936082072979, 9.176010335696908, 11.987009617447153, 1.0488993643373625, -3.744098587161166, 57.80395980897628]
        if self.color == COLOR_BLACK:
            self.chessboard_weight = assign_weight_array(self.black_weight_vector, self.chessboard_size)
        else:
            self.chessboard_weight = assign_weight_array(self.white_weight_vector, self.chessboard_size)

    def from_list(self, arg_list):
        self.weight_vector = arg_list[:10]
        self.chessboard_weight = assign_weight_array(self.weight_vector, self.chessboard_size)

    def to_list(self):
        return str(list(self.weight_vector))

    # The input is current chessboard.
    def go(self, chessboard):
        self.candidate_list.clear()
        self.chessboard = chessboard
        self.candidate_list = generateCandidate(chessboard, self.color)
        move = self.absearch()
        if move:
            self.candidate_list.append(move)

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
        childlist = []
        for c in child_list:
            childlist.append(c[1])
        return childlist

    def absearch(self):
        alpha = -infinity
        beta = infinity
        stage = np.count_nonzero(self.chessboard)
        start = time.time()
        child_list = self.presearch()
        end = time.time()
        presearch_time = end - start
        if len(child_list) != 0:
            self.candidate_list.append(child_list[0])
        if stage < 50:
            level = 6
            move, _ = self.minimize(self.chessboard, alpha, beta, level, child_list)
        else:
            level = 8
            move, _ = self.minimize(self.chessboard, alpha, beta, level, child_list)
        return move

    def maximize(self, chessboard, alpha, beta, level):
        if level == 0:
            return None, self.evaluate(chessboard, self.color)
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

    def minimize(self, chessboard, alpha, beta, level, childlist = None):
        if level == 0:
            return None, self.evaluate(chessboard, self.color)
        minChild = None
        minUtility = infinity
        if childlist:
            children = childlist
        else:
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
        chessboard[(i_arr, j_arr)] = color
        chessboard[move] = color
        return i_arr, j_arr

    def undo_place(self, update_i, update_j, move, chessboard, color):
        chessboard[(update_i, update_j)] = -color
        chessboard[move[0]][move[1]] = COLOR_NONE

    def evaluate(self, chessboard, color):
        idx1 = np.where(chessboard == color)  # 自己
        idx2 = np.where(chessboard == -color)  # 对手
        res = self.chessboard_weight[idx1].sum() - self.chessboard_weight[idx2].sum()  # 自己减对手的差
        return res


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


def assign_weight_array(v, csize):
    assert csize == 8

    weight_matrix = np.array([
        [v[9], v[8], v[6], v[3], v[3], v[6], v[8], v[9]],
        [v[8], v[7], v[5], v[2], v[2], v[5], v[7], v[8]],
        [v[6], v[5], v[4], v[1], v[1], v[4], v[5], v[6]],
        [v[3], v[2], v[1], v[0], v[0], v[1], v[2], v[3]],
        [v[3], v[2], v[1], v[0], v[0], v[1], v[2], v[3]],
        [v[6], v[5], v[4], v[1], v[1], v[4], v[5], v[6]],
        [v[8], v[7], v[5], v[2], v[2], v[5], v[7], v[8]],
        [v[9], v[8], v[6], v[3], v[3], v[6], v[8], v[9]]
    ])
    return weight_matrix



