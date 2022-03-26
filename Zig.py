import numpy as np

def zigtable():
    # Zig Order Table For Looking Up
    table = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), \
            (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), \
            (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), \
            (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), \
            (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), \
            (5, 5), (4, 6), (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), \
            (6, 7), (7, 6), (7, 7)]

    # save
    np.savetxt('./ref/zig_table.txt', table)

def zigscan(table, M):
    m = np.array([M[i, j] for i, j in table])
    return m

def inv_zigscan(table, m):
    M = np.zeros((8, 8))
    for i in range(len(table)):
        row, col = table[i][0], table[i][1]
        M[row, col] = m[i]
    return M

if __name__ == '__main__':
    #zigtable()
    M = np.arange(64).reshape((8, 8))
    print(M)
    table = np.loadtxt('./ref/zig_table.txt').astype(np.uint8)
    m = zigscan(table, M)
    print(m)
    M = inv_zigscan(table, m)
    print(M)