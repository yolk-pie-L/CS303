import numpy as np
from main import AI

ai = AI(8, 1, 5)

chessborad = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, -1, 0, 0, 0],
                       [0, 0, 0, -1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

# k = list(zip(idx[0], idx[1]))
# print(k)
ai.go(chessborad)
print(ai.candidate_list)