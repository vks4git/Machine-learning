import math
from random import uniform


class Point:
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f

    def __lt__(self, other):
        return self.f < other.f


def genetic_alg(f, a, b, pop_size):
    init = []
    pop_size = max(pop_size, 2)
    for i in range(pop_size):
        x = uniform(a[0], b[0])
        y = uniform(a[1], b[1])
        init.append(Point(x, y, f([x, y])))
    for iter in range(100):
        for i in range(pop_size ** 2):
            p1 = int(uniform(0, pop_size))
            p2 = int(uniform(0, pop_size))
            while p1 == p2:
                p2 = int(uniform(0, pop_size))
            a = init[p1].x
            b = init[p2].x
            if a > b:
                a, b = b, a
            nx = uniform(a, b)
            a = init[p1].y
            b = init[p2].y
            if a > b:
                a, b = b, a
            ny = uniform(a, b)
            init.append(Point(nx, ny, f([nx, ny])))
        init = sorted(init)
        groups = max(1, int(math.sqrt(len(init))))
        dict = [2 ** 256] * groups
        indices = [2 ** 256] * groups
        for i in range(len(init)):
            ind = int(uniform(0, groups))
            if dict[ind] > init[i].f:
                dict[ind] = init[i].f
                indices[ind] = i
        new_pop = [init[i] for i in indices if i < 2 ** 256]
        init = new_pop[::]
        pop_size = len(init)
        if pop_size < 2:
            break
    best_val = 2 ** 256
    best_pt = None
    for i in init:
        if i.f < best_val:
            best_val = i.f
            best_pt = i
    return best_pt


def nice_func(x):
    return (x[0] - 4) ** 2 + (x[1] + 2) ** 2 - 1


def weird_func(x):
    return math.e ** ((x[0] + x[1] - 4) / 100.0) * (math.sin(x[0]) + math.cos(x[1])) + math.e ** (x[0] + x[1] + 2)


def banana_func(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
