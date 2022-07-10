import numpy as np
import random
import time
import math

infinity = math.inf

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

directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def eval(chessboard, color):
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
        self.chessboard = chessboard
        self.candidate_list = self.generateCandidate(chessboard, self.color)
        move = self.absearch()
        if move:
            self.candidate_list.append(move)

    def absearch(self):
        alpha = -infinity
        beta = infinity
        level = 4
        move, _ = self.minimize(self.chessboard, alpha, beta, level)
        return move

    def maximize(self, chessboard, alpha, beta, level):
        if level == 0:
            return None, eval(chessboard, self.color)
        maxChild = None
        maxUtility = -infinity
        children = self.generateCandidate(chessboard, -self.color)
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
            return None, eval(chessboard, self.color)
        minChild = None
        minUtility = infinity
        children = self.generateCandidate(chessboard, self.color)
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

    # generate candidate list
    def generateCandidate(self, chessboard, color):
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
                while 0 <= i_tempt + direction[0] < self.chessboard_size and \
                        0 <= j_tempt + direction[1] < self.chessboard_size and \
                        chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == -color:
                    i_tempt = i_tempt + direction[0]
                    j_tempt = j_tempt + direction[1]
                    moved = True
                if moved and 0 <= i_tempt + direction[0] < self.chessboard_size and \
                        0 <= j_tempt + direction[1] < self.chessboard_size and \
                        chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == color:
                    candidate_list.append((i, j))
                    break
        return candidate_list

    def place(self, chessboard, move, color):
        i_arr = []
        j_arr = []
        for direction in directions:
            i_tempt = move[0]
            j_tempt = move[1]
            moved = False
            i_tempt_arr = []
            j_tempt_arr = []
            while 0 <= i_tempt + direction[0] < self.chessboard_size and \
                    0 <= j_tempt + direction[1] < self.chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == -color:
                i_tempt = i_tempt + direction[0]
                j_tempt = j_tempt + direction[1]
                moved = True
                i_tempt_arr.append(i_tempt)
                j_tempt_arr.append(j_tempt)
            if moved and 0 <= i_tempt + direction[0] < self.chessboard_size and \
                    0 <= j_tempt + direction[1] < self.chessboard_size and \
                    chessboard[i_tempt + direction[0]][j_tempt + direction[1]] == color:
                i_arr = i_arr + i_tempt_arr
                j_arr = j_arr + j_tempt_arr
        chessboard[(i_arr,j_arr)] = color
        chessboard[move] = color
        return i_arr, j_arr

    def undo_place(self, update_i, update_j, move, chessboard, color):
        chessboard[(update_i, update_j)] = -color
        chessboard[move[0]][move[1]] = COLOR_NONE


# ==============Find new pos========================================
# Make sure that the position of your decision in chess board is empty.
# If not, the system will return error.
# Add your decision into candidate_list, Records the chess board
# You need add all the positions which is valid
# candidate_list example: [(3,3),(4,4)]
# You need append your decision at the end of the candidate_list,
# we will pickthe last element of the candidate_list as the position you choose
# If there is no valid position, you must return an empty list.

ai = AI(8, -1, 5)

chessborad = np.array([
    [0, -1, -1, -1, -1, -1, -1, 0], [0, 1, 1, -1, 1, 1, 1, -1], [0, 0, 1, -1, -1, 1, 1, -1], [0, 1, 1, 1, 1, 1, 1, -1],
    [0, 0, 1, -1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

# k = list(zip(idx[0], idx[1]))
# print(k)
ai.go(chessborad)
print(ai.candidate_list)
