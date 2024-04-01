import numpy as np

def extract_square(img, nrow, ncol):
    assert len(img.shape) == 2, "Image must be grayscale"
    nr, nc = img.shape
    assert nr >= nrow and nc >= ncol, "Image too small"
    rmarg = (nr-nrow)//2
    cmarg = (nc-ncol)//2
    return img[rmarg:rmarg+nrow, cmarg:cmarg+ncol] # return middle square


class Graph(object):
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.vertices = np.arange(1, n_vertices + 1)
        self.edges = edges  # adjacency list implemented as dictionary mapping vertex (integer 1,...,n_vertices) to list of neighbors
        self.L = None
        self.A = None
        self.D = None

    def get_laplacian_adjacency_degree_matrix(self):
        if self.L is None:
            n = self.n_vertices
            D = np.zeros((n,))
            A = np.zeros((n, n))
            for i in np.arange(n):
                neighbors = self.edges.get(i + 1, [])
                #print("node", i + 1, "neighbors", neighbors)
                for nbr, wgt in neighbors:
                    A[i, nbr - 1] = wgt
                    D[i] += wgt
            self.A = A
            self.D = np.diag(D)
            self.L = self.D - self.A
        return self.L, self.A, self.D

    def __str__(self):
        return f"{self.n_vertices}__{self.edges}"


class Image(Graph):
    def __init__(self, M, N):
        n_vertices = M * N
        edges = {}
        for i in range(M):
            for j in range(N):
                node = i * M + j + 1
                left = i * M + (j - 1) % N + 1
                right = i * M + (j + 1) % N + 1
                top = ((i - 1) % M) * M + j + 1
                bottom = ((i + 1) % M) * M + j + 1
                edges[node] = [(left, 1), (right, 1), (top, 1), (bottom, 1)]
        super().__init__(n_vertices, edges)
