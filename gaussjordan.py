"""
Author: Shrehan Raj Singh

Solving
x + y + z = 6
2x + 2y + z = 9
x + 2y + 2z = 12
"""

A = [
    [1, 1, 1],
    [2, 2, 1],
    [1, 2, 2]
]

B = [[6], [9], [12]]

aug = [
    x[0] + x[1] for x in zip(A, B)
]


def gauss_jordan(A):
    n = len(A)
    m = len(A[0])

    row = 0
    for col in range(m):
        pivot = row
        for j in range(row, n):
            if A[j][col] != 0:
                pivot = j
                break

        if j == n:
            continue

        A[pivot], A[row] = A[row], A[pivot]

        # make pivot 1
        f = A[row][col]
        A[row] = [x / f for x in A[row]]

        # make all elements in the column 0
        for i in range(n):
            if i == row:
                continue

            f = A[i][col]
            rm = [x * f for x in A[row]]

            for j in range(m):
                A[i][j] -= rm[j]

        row += 1
        if row == n:
            break

    return A


print(gauss_jordan(aug))
