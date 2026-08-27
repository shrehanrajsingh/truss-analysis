"""
Author: Shrehan Raj Singh
"""

from typing import List
import matplotlib.pyplot as plt

# CONSTANTS
A = 500
E = 200000
AE = A * E


class Joint:
    id: int = 0  # joint number
    x: int = 0
    y: int = 0
    dc: List[int] = None  # DOF is constrained? 0 -> no, 1 -> yes (x, y)

    def __init__(self, id=0, x=0, y=0, dc=None):
        self.id = id
        self.x = x
        self.y = y
        self.dc = dc if dc else [0, 0]

    def __eq__(self, value):
        if isinstance(value, Joint):
            return self.id == value.id
        return False


class Member:
    L: int = 0
    j: List[Joint] = None

    def __init__(self, L=0, j=None):
        self.L = L
        self.j = j

    def _calc_cs(self):
        assert len(self.j) == 2, "member not connected to joints"

        dx = self.j[1].x - self.j[0].x
        dy = self.j[1].y - self.j[0].y

        return dx / self.L, dy / self.L

    def make_K(self):
        c, s = self._calc_cs()
        c2 = round(c * c, 6) * AE / self.L
        s2 = round(s * s, 6) * AE / self.L
        cs = round(c * s, 6) * AE / self.L

        return [
            [c2, cs, -c2, -cs],
            [cs, s2, -cs, -s2],
            [-c2, -cs, c2, cs],
            [-cs, -s2, cs, s2]
        ]


truss = [
    Member(400, [
        Joint(1, 0, 0, [1, 1]),
        Joint(2, 400, 0, [0, 1])
    ]),
    Member(300, [
        Joint(2, 400, 0, [0, 1]),
        Joint(3, 400, 300)
    ]),
    Member(500, [
        Joint(1, 0, 0, [1, 1]),
        Joint(3, 400, 300)
    ])
]


def make_gK(n):
    """
    n: number of joints
    returns: matrix of size 2n x 2n with 0s
    """

    return [[0 for _ in range(2 * n)] for _ in range(2 * n)]


def add_lK_to_K(K, lK, p):
    """
    K: global K
    lK: K of one member
    p: position coordinates
    """

    cm = [
        [(p[0], p[0]), (p[0], p[1]), (p[0], p[2]), (p[0], p[3])],
        [(p[1], p[0]), (p[1], p[1]), (p[1], p[2]), (p[1], p[3])],
        [(p[2], p[0]), (p[2], p[1]), (p[2], p[2]), (p[2], p[3])],
        [(p[3], p[0]), (p[3], p[1]), (p[3], p[2]), (p[3], p[3])]
    ]

    n = len(lK)
    m = len(lK[0])

    for i in range(n):
        for j in range(m):
            ii, jj = cm[i][j]

            # 1-indexed -> 0-indexed
            ii -= 1
            jj -= 1
            K[ii][jj] += lK[i][j]  # superposition


K = make_gK(3)

for i, iv in enumerate(truss):
    lK = iv.make_K()

    add_lK_to_K(K, lK, [iv.j[0].id * 2 - 1, iv.j[0].id *
                2, iv.j[1].id * 2 - 1, iv.j[1].id * 2])

F = [[0],
     [0],
     [0],
     [0],
     [0],
     [-10]]

joint_map = {}
for i in truss:
    if i.j[0].id not in joint_map:
        joint_map[i.j[0].id] = i.j[0]

    if i.j[1].id not in joint_map:
        joint_map[i.j[1].id] = i.j[1]

U = [1 for _ in range(2 * len(joint_map))]  # all required to find

for i in joint_map:
    iv: Joint = joint_map[i]

    if iv.dc[0]:
        U[(i * 2 - 1) - 1] = 0  # constrained, remove

    if iv.dc[1]:
        U[(i * 2) - 1] = 0  # constrained, remove

# Now, we find those entries for which we have U != 0
idxs = [x for x in range(len(U)) if U[x]]

aug = []
for i in idxs:
    row = []

    for j in idxs:
        row.append(K[i][j])

    row.append(F[i][0])
    aug.append(row)

# Now, we do a Gauss Jordan over `aug`


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

        A[row], A[pivot] = A[pivot], A[row]

        f = A[row][col]
        A[row] = [x / f for x in A[row]]

        for i in range(n):
            if i == row:
                continue

            f = A[i][col]
            for j in range(m):
                A[i][j] -= f * A[row][j]

        row += 1
        if row == n:
            break

    return A


r = gauss_jordan(aug)

U_sol = [0 for _ in range(2 * len(joint_map))]

c = 0
for i in idxs:
    U_sol[i] = r[c][-1]
    c += 1

# original members
for i in truss:
    x = [i.j[0].x, i.j[1].x]
    y = [i.j[0].y, i.j[1].y]

    plt.plot(x, y, marker='o', linestyle='--')

scale = 1e5
# deformed members
for i in truss:
    x = []
    y = []

    for j in i.j:
        dof_x = 2 * (j.id - 1)
        dof_y = dof_x + 1

        ux = U_sol[dof_x]
        uy = U_sol[dof_y]

        nx = scale * ux + j.x
        ny = scale * uy + j.y

        x.append(nx)
        y.append(ny)

    plt.plot(x, y, marker='o', linestyle='--')

plt.axis('equal')
plt.show()
