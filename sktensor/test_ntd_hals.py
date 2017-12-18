import numpy as np

from sktensor import tucker
from sktensor.dtensor import dtensor, unfolded_dtensor

T = np.zeros((3, 4, 5))
T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
T[:, :, 2] = [[3, 16, 9, 22], [1, 17, 2, 2], [1, 1, 1, 2]]
T[:, :, 3] = [[3, 16, 9, 22], [1, 7, 12, 33], [11, 31, 41, 2]]
T[:, :, 4] = [[15, 16, 9, 2], [1, 17, 2, 2], [51, 51, 31, 2]]
T = dtensor(T)

print('Tensor shape:')
print(T.shape)

ranks = [2, 3, 2]
print('ranks:')
print(np.asarray(ranks))

A = tucker.ntd_hals(T, ranks)

print('Component matrices:')
print('A[0]:', np.sum(A[0], axis=0))
print('A[1]:', np.sum(A[1], axis=0))
print('A[2]:', np.sum(A[2], axis=0))
