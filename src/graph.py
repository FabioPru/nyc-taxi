import random
import numpy as np
from functools import lru_cache

from .nyc import nyc_N, nyc_vertex_pairs

class Graph():

    def __init__(self, N, edges):
        """N: number of nodes
        edges: list of pairs of adjacent vertices
        Graph(4, [(0, 1), (2, 3)])
        """
        self.N = N
        self.M = len(edges)
        self.edges = edges

        self.a = [[] for _ in range(N)]
        for x, y in edges:
            self.a[x].append(y)
            self.a[y].append(x)

        self.d = {}
        self.avgd = 0.0
        for x in range(self.N):
            for y in range(self.N):
                self.d[x, y] = self.bfs(x, y)
                self.avgd += self.d[x, y]
        self.avgd = self.avgd / (self.N ** 2)

    @lru_cache(maxsize=8192)
    def bfs(self, v1, v2):
        q = [(v1, 0)]
        visited = {i: False for i in range(self.N)}

        while len(q) > 0:
            v, d = q[0]
            q = q[1:]

            if v == v2:
                return d

            if not visited[v]:
                visited[v] = True
                for adj in self.a[v]:
                    q.append((adj, d + 1))

        print("Careful: graph not connected", v1, v2)
        return np.inf

    def rand_v(self):
        return random.randint(0, self.N - 1)


# Small

# 0 - 1
# | /
# 2
# |
# 3

G = {'small': Graph(4, [(0, 1), (1, 2), (0, 2), (2, 3)]),
     'medium': Graph(20, [(random.randint(0, 19), random.randint(0, 19)) for _ in range(100)]),
     'large': Graph(100, [(random.randint(0, 99), random.randint(0, 99)) for _ in range(400)]),
     'line': Graph(20, [(i, i+1) for i in range(19)]),
     'complete': Graph(20, [(i, j) for i in range(20) for j in range(i+1, 20)]),
     'nyc': Graph(nyc_N, nyc_vertex_pairs)}
