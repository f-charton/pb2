from envs.environment import DataPoint, BaseEnvironment
import math
import numpy as np
from utils import bool_flag
from .tokenizers import SparseTokenizer, DenseTokenizer, EdgeTokenizer
from utils import bool_flag



class CycleDataPoint(DataPoint):
    N = 4
    HARD = True

    def __init__(self, init=False):
        super().__init__()
        self.matrix = np.zeros((self.N, self.N), dtype=np.int32)
        self.cycles = []
        if init:
            self._add_edges_greedily()
            self.calc_score()
            self.calc_features()

    def calc_score(self):
        if self.HARD and len(self.cycles) > 0:
            self.score = -1
            return
        self.score = self.matrix.sum().item() // 2 - self.PENALTY * len(self.cycles)

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.matrix[i, j])
        self.features = ",".join(map(str, w))

    def _add_edges_greedily(self):
        if self.TASK == "3cycles":
            adjmat_cycle = self.matrix @ self.matrix
        elif self.TASK == "4cycles":
            adjmat_cycle = self.matrix @ self.matrix @ self.matrix
        else:
            raise ValueError(f"Invalid task: {self.TASK}")
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.matrix[i, j] == 0 and adjmat_cycle[i, j] == 0:
                    allowed_edges.append((i, j))

        while allowed_edges:
            i, j = allowed_edges[np.random.randint(len(allowed_edges))]
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1
            new_allowed_edges = []
            if self.TASK == "3cycles":
                adjmat_cycle = self.matrix @ self.matrix
            elif self.TASK == "4cycles":
                adjmat_cycle = self.matrix @ self.matrix @ self.matrix
            for a, b in allowed_edges:
                if self.matrix[a, b] == 0 and adjmat_cycle[a, b] == 0:
                    new_allowed_edges.append((a, b))
            allowed_edges = new_allowed_edges

    def _remove_edges_greedily(self):
        while self.cycles:
            edge_count = {}
            for cycle in self.cycles:
                for edge in cycle:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            most_frequent_edge = max(edge_count, key=edge_count.get)
            i, j = most_frequent_edge
            self.matrix[i, j] = 0
            self.matrix[j, i] = 0

            remaining_cycles = []
            for cycle in self.cycles:
                if (i, j) not in cycle:
                    remaining_cycles.append(cycle)
            self.cycles = remaining_cycles

    def _cycles_computation(self):
        return

    def local_search(self):
        self._cycles_computation()
        self._remove_edges_greedily()
        self._add_edges_greedily()
        self._cycles_computation()
        self.calc_score()
        self.calc_features()

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


class TriangleDataPoint(CycleDataPoint):
    TASK = "3cycles"
    PENALTY = 2

    def _cycles_computation(self):
        cycles = set()
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
                if self.matrix[i, j] == 1:
                    common = bits_i & row_bits[j]
                    x = common
                    while x:
                        lsb_u = x & -x
                        u = lsb_u.bit_length() - 1
                        x ^= lsb_u
                        elems = sorted([i, u, j])
                        a, b, c = elems
                        cycles.add((a, b, c))
        
        self.cycles = []
        for cycle in cycles:
            a, b, c = cycle
            self.cycles.append(((a, b), (b, c), (a, c)))


class SquareDataPoint(CycleDataPoint):
    TASK = "4cycles"
    PENALTY = 6

    def _cycles_computation(self):
        cycles = set()
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
                        cycles.add((a, b, c, d))

        self.cycles = []
        for cycle in cycles:
            a, b, c, d = cycle
            self.cycles.append(((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, d), max(c, d)), (min(d, a), max(d, a))))


class CycleEnvironment(BaseEnvironment):
    def __init__(self, params):
        super().__init__(params)
        self.data_class.N = params.N
        self.data_class.HARD = params.hard
        if params.encoding_tokens == "edge_single_token":
            base = params.N * (params.N - 1) // 2
            self.tokenizer = SparseTokenizer(params.N, self.data_class)
            self.symbols = [str(i) for i in range(base)]
        elif params.encoding_tokens == "edge_double_tokens":
            base = params.N
            self.tokenizer = EdgeTokenizer(params.N, self.data_class,params.nosep)
            self.symbols = [str(i) for i in range(base)]
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(params.N, self.data_class,params.nosep)
            self.symbols = [str(i) for i in range(2)]
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

        self.symbols.extend(BaseEnvironment.SPECIAL_SYMBOLS)

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--N', type=int, default=30, help='Number of vertices in the K-cycle-free graph')
        parser.add_argument('--hard', type=bool_flag, default="true", help='whether only K-cycle-free graphs are accepted')
        parser.add_argument('--encoding_tokens', type=str, default="edge_single_token", help='toknized by edge or adjacency matrix')
        parser.add_argument('--nosep', type=bool_flag, default="false", help='separator (for adjacency and double edge)')


class SquareEnvironment(CycleEnvironment):
    data_class = SquareDataPoint
    def __init__(self, params):
        super().__init__(params)


class TriangleEnvironment(CycleEnvironment):
    data_class = TriangleDataPoint
    def __init__(self, params):
        super().__init__(params)
