from envs.environment import DataPoint, BaseEnvironment
import math
import random
import numpy as np
from itertools import permutations
from utils import bool_flag
from .tokenizers import SparseTokenizer, DenseTokenizer
from typing import Optional, List, Tuple



class SquareDataPoint(DataPoint):
    N = 4
    SQUARE_HARD = True
    INIT_METHOD = 'edge_addition'

    def __init__(self, val=None):
        super().__init__()
        self.val = val
        if val is None:
            self.val = []
            if self.INIT_METHOD == "edge_removal":
                self._generate_graph_by_edge_removal()
            elif self.INIT_METHOD == "edge_addition":
                self._generate_graph_by_edge_addition()
        self.calc_features()
        self.calc_score()
        
    @classmethod
    def _update_class_params(self,pars):
        self.N = pars[0]
        self.SQUARE_HARD = pars[1]
        self.INIT_METHOD = pars[2]

    @classmethod
    def _save_class_params(self):
        return (self.N, self.SQUARE_HARD, self.INIT_METHOD)

    def _edge_to_index(self, i: int, j: int) -> int:
        """Convert edge (i,j) to linear index in upper triangular matrix"""
        return i * (2 * self.N - i - 1) // 2 + (j - i - 1)

    def _index_to_edge(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to edge (i,j) in upper triangular matrix"""
        discriminant = (2 * self.N - 1) ** 2 - 8 * idx
        if discriminant < 0:
            i = self.N - 2
        else:
            i = int((2 * self.N - 1 - math.sqrt(discriminant)) // 2)
            i = max(0, min(i, self.N - 2))
        
        Sx = i * (2 * self.N - i - 1) // 2
        j = i + 1 + idx - Sx
        
        return (i, j)

    def _ensure_unique_val(self) -> None:
        """Ensure self.val contains only unique elements and is sorted"""
        self.val = sorted(list(set(self.val)))

    def _generate_graph_by_edge_removal(self) -> List[int]:
        """Generate random edges for initialization"""
        max_edges = self.N * (self.N - 1) // 2
        num_edges = random.randint(0, max_edges)
        all_edges = list(range(max_edges))
        random.shuffle(all_edges)

        self.val = sorted(all_edges[:num_edges])
        self.calc_features()
        self._remove_squares_greedily()

    def _generate_graph_by_edge_addition(self) -> List[int]:
        self.val = []
        self.calc_features()
        self._add_edges_greedily()

    def _create_matrix(self):
        self.matrix = np.zeros((self.N, self.N), dtype=int)
        for v in self.val:
            i, j = self._index_to_edge(v)
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1

    def _squares(self):
        row_bits = []
        for i in range(self.N):
            mask = 0
            row = self.matrix[i]
            for j in range(self.N):
                if row[j] == 1:
                    mask |= 1 << j
            row_bits.append(mask)
        
        self.squares = []
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
                        self.squares.append((i, u, j, v))


    def calc_score(self):
        if self.SQUARE_HARD and len(self.squares) > 0:
            self.score = -1
            return
        self.score = len(self.val) - 6 * len(self.squares)

    def calc_features(self):
        """
        Compute optional features to give more signal to the transformer
        """
        self._ensure_unique_val()
        self._create_matrix()
        self._squares()
        
    
    def local_search(self) -> None:
        self._remove_squares_greedily()
        self._add_edges_greedily()
        self._squares()
        self.calc_score()

    def _get_square_edges(self, vertices: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        i, j, k, l = vertices
        for v1, v2, v3 in permutations([j, k, l]):
            if (self.matrix[i, v1] == 1 and 
                self.matrix[v1, v2] == 1 and 
                self.matrix[v2, v3] == 1 and 
                self.matrix[v3, i] == 1):
                edges = [(min(i, v1), max(i, v1)), 
                         (min(v1, v2), max(v1, v2)),
                         (min(v2, v3), max(v2, v3)),
                         (min(v3, i), max(v3, i))]
                return edges

    def _remove_squares_greedily(self) -> None:
        while self.squares:
            edge_count = {}
            for square in self.squares:
                edges = self._get_square_edges(square)
                if edges is not None:
                    for edge in edges:
                        edge_count[edge] = edge_count.get(edge, 0) + 1
            
            most_frequent_edge = max(edge_count, key=edge_count.get)
            
            i, j = most_frequent_edge
            edge_idx = self._edge_to_index(i, j)
            if edge_idx in self.val:
                self.val.remove(edge_idx)
                self.matrix[i, j] = 0
                self.matrix[j, i] = 0

            remaining_squares = []
            for square in self.squares:
                edges = self._get_square_edges(square)
                if edges is not None and most_frequent_edge not in edges:
                    remaining_squares.append(square)
            self.squares = remaining_squares

    def _add_edges_greedily(self) -> None:
        adjmat3 = self.matrix @ self.matrix @ self.matrix
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.matrix[i, j] == 0 and adjmat3[i, j] == 0:
                    allowed_edges.append((i, j))
        
        while allowed_edges:
            edge = random.choice(allowed_edges)
            i, j = edge
            
            edge_idx = self._edge_to_index(i, j)
            if edge_idx not in self.val:
                self.val.append(edge_idx)
                self.val.sort()
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1
            
            new_allowed_edges = []
            adjmat3 = self.matrix @ self.matrix @ self.matrix
            for (a, b) in allowed_edges:
                if self.matrix[a, b] == 0 and adjmat3[a, b] == 0:
                    new_allowed_edges.append((a, b))
            
            allowed_edges = new_allowed_edges
        
        self._ensure_unique_val()



class SquareEnvironment(BaseEnvironment):
    data_class = SquareDataPoint
    def __init__(self, params):
        super().__init__(params)
        SquareDataPoint.N = params.square_N
        SquareDataPoint.SQUARE_HARD = params.square_hard
        SquareDataPoint.INIT_METHOD = params.square_init_method
        if params.edge_tokens:
            base = params.square_N * (params.square_N - 1) // 2
            self.tokenizer = SparseTokenizer(params.square_N, SquareDataPoint)
            self.symbols = [str(i) for i in range(base)]
        else:
            self.tokenizer = DenseTokenizer(params.square_N, SquareDataPoint)
            self.symbols = [str(i) for i in range(3)]

        self.symbols.extend(params.symbols.split(","))
        
    #def generate_data(self, size):
    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--square_N', type=int, default=30, help='Number of vertices in the square-free graph')
        parser.add_argument('--square_hard', type=bool_flag, default="false", help='whether only square-free graphs are accepted')
        parser.add_argument('--square_init_method', type=str, default="edge_removal", help='method of generation')
        parser.add_argument('--edge_tokens', type=bool_flag, default="false", help='toknized by edge or adjacency matrix')

    #def do_score(self)

