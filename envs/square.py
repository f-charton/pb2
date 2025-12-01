from envs.environment import DataPoint, BaseEnvironment
import math
import numpy as np
from utils import bool_flag
from .tokenizers import SparseTokenizer, DenseTokenizer



class SquareDataPoint(DataPoint):
    N = 4
    HARD = True

    def __init__(self, init=False):
        super().__init__()
        self.matrix = np.zeros((self.N, self.N), dtype=int)
        self.squares = []
        if init:
            self._add_edges_greedily()
            self.calc_score()

    def calc_score(self):
        if self.HARD and len(self.squares) > 0:
            self.score = -1
            return
        self.score = self.matrix.sum().item() // 2 - 6 * len(self.squares)

    def calc_features(self):
        pass

    def _add_edges_greedily(self):
        adjmat3 = self.matrix @ self.matrix @ self.matrix
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.matrix[i, j] == 0 and adjmat3[i, j] == 0:
                    allowed_edges.append((i, j))

        while allowed_edges:
            i, j = allowed_edges[np.random.randint(len(allowed_edges))]
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1
            new_allowed_edges = []
            adjmat3 = self.matrix @ self.matrix @ self.matrix
            for a, b in allowed_edges:
                if self.matrix[a, b] == 0 and adjmat3[a, b] == 0:
                    new_allowed_edges.append((a, b))
            allowed_edges = new_allowed_edges

    def _remove_edges_greedily(self):
        while self.squares:
            edge_count = {}
            for square in self.squares:
                for edge in square:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            most_frequent_edge = max(edge_count, key=edge_count.get)
            i, j = most_frequent_edge
            self.matrix[i, j] = 0
            self.matrix[j, i] = 0

            remaining_squares = []
            for square in self.squares:
                if (i, j) not in square:
                    remaining_squares.append(square)
            self.squares = remaining_squares

    def _squares_computation(self):
        squares = set()
        row_bits = []
        for i in range(self.N):
            mask = 0
            for j in range(self.N):
                if self.matrix[i, j] == 1:
                    mask |= 1 << j
            row_bits.append(mask)

        for i in range(self.N):
            bits_i = row_bits[i]
            for j in range(i + 1, self.N):
                common = bits_i & row_bits[j]
                x = common
                while x:
                    lsb_u = x & -x
                    u = lsb_u.bit_length() - 1
                    x ^= lsb_u
                    y = x
                    while y:
                        lsb_v = y & -y
                        v = lsb_v.bit_length() - 1
                        y ^= lsb_v

                        # add unique sorting to speed up the future computations
                        elems = [i, u, j, v]
                        a = min(elems)
                        min_idx = elems.index(a)
                        neighbours = [elems[(min_idx + 1) % 4], elems[(min_idx - 1) % 4]]
                        b = min(neighbours)
                        d = max(neighbours)
                        c = sum(elems) - a - b - d
                        squares.add((a, b, c, d))

        self.squares = []
        for square in squares:
            a, b, c, d = square
            self.squares.append(((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, d), max(c, d)), (min(d, a), max(d, a))))

    def local_search(self):
        self._squares_computation()
        self._remove_edges_greedily()
        self._add_edges_greedily()
        self._squares_computation()
        self.calc_score()

    @classmethod
    def _update_class_params(self,pars):
        self.N = pars[0]
        self.HARD = pars[1]

    @classmethod
    def _save_class_params(self):
        return (self.N, self.HARD)

    @classmethod
    def _batch_generate_and_score(cls,n, pars=None):
        return super()._batch_generate_and_score(n,pars)


class SquareEnvironment(BaseEnvironment):
    data_class = SquareDataPoint
    def __init__(self, params):
        super().__init__(params)
        SquareDataPoint.N = params.square_N
        SquareDataPoint.HARD = params.square_hard
        if params.encoding_tokens == "edge":
            base = params.square_N * (params.square_N - 1) // 2
            self.tokenizer = SparseTokenizer(params.square_N, SquareDataPoint)
            self.symbols = [str(i) for i in range(base)]
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(params.square_N, SquareDataPoint)
            self.symbols = [str(i) for i in range(2)]
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

        self.symbols.extend(params.symbols.split(","))
        
    #def generate_data(self, size):
    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--square_N', type=int, default=30, help='Number of vertices in the square-free graph')
        parser.add_argument('--square_hard', type=bool_flag, default="false", help='whether only square-free graphs are accepted')
        parser.add_argument('--encoding_tokens', type=str, default="edge", help='toknized by edge or adjacency matrix')


