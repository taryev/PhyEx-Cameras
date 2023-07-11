import numpy as np


def dtw_distance(data1, data2):
    m = len(data1)
    n = len(data2)

    cost_matrix = np.zeros((m, n))

    cost_matrix[0, 0] = abs(data1[0] - data2[0])
    for i in range(1, m):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + abs(data1[i] - data2[0])
    for j in range(1, n):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + abs(data1[0] - data2[j])

    for i in range(1, m):
        for j in range(1, n):
            cost_matrix[i, j] = abs(data1[i] - data2[j]) + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1],
                                                               cost_matrix[i - 1, j - 1])

    return cost_matrix[m - 1, n - 1]