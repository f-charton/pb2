from envs.environment import DataPoint, BaseEnvironment
from .utils import sort_graph_based_on_degree, random_symmetry_adj_matrix
import numpy as np
from utils import bool_flag
from .tokenizers import SparseTokenizer, DenseTokenizer


class CycleDataPoint(DataPoint):
    N = 4
    HARD = True
    MAKE_OBJECT_CANONICAL = False
    PENALTY = 6
    BALANCED = False

    def __init__(self, init=False):
        super().__init__()
        self.matrix = np.zeros((self.N, self.N), dtype=np.uint8)
        self.cycles = []

        if init:
            self._add_edges_greedily()
            self.calc_score()
            if self.MAKE_OBJECT_CANONICAL:
                self.matrix = sort_graph_based_on_degree(self.matrix)
            self.calc_features()
            self.calc_score()

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
        self._cycles_computation()

    def _add_edges_greedily(self):
        np.random.seed(None)
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
            if self.BALANCED:
                cycle_edges = []
                for cycle in self.cycles:
                    cycle_edges.extend(cycle)
                unique_cycle_edges = set(cycle_edges)
                selected_edge = np.random.choice(list(unique_cycle_edges))
            else:
                edge_count = {}
                for cycle in self.cycles:
                    for edge in cycle:
                        edge_count[edge] = edge_count.get(edge, 0) + 1
                selected_edge = max(edge_count, key=edge_count.get)
            i, j = selected_edge
            self.matrix[i, j] = 0
            self.matrix[j, i] = 0

            remaining_cycles = []
            for cycle in self.cycles:
                if (i, j) not in cycle:
                    remaining_cycles.append(cycle)
            self.cycles = remaining_cycles

    def _cycles_computation(self):
        return
    
    def mutate(self, n):
        for _ in range(n):
            i = np.random.randint(1,self.N)
            j = np.random.randint(i)
            self.matrix[i][j]=1
            self.matrix[j][i]=1
        self.local_search()


    def local_search(self):
        #self._cycles_computation()
        self._remove_edges_greedily()
        self._add_edges_greedily()
        self._cycles_computation()
        self.calc_score()
        if self.MAKE_OBJECT_CANONICAL:
            self.matrix = sort_graph_based_on_degree(self.matrix)
        self.calc_features()
        self.calc_score()

    def redeem(self):
        self._remove_edges_greedily()
        self.calc_features()
        self.calc_score()


    @classmethod
    def _update_class_params(self,pars):
        self.N = pars[0]
        self.HARD = pars[1]
        self.MAKE_OBJECT_CANONICAL = pars[2]
        self.PENALTY = pars[3]
        self.BALANCED = pars[4]
        self.TASK = pars[5]

    @classmethod
    def _save_class_params(self):
        return (self.N, self.HARD, self.MAKE_OBJECT_CANONICAL, self.PENALTY, self.BALANCED,self.TASK)


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
    # this problem lives in N^2, so we can use k=2
    # this problem is symmetric, so we can use is_adj_matrix_symmetric=True
    k = 2
    is_adj_matrix_symmetric = True
    def __init__(self, params):
        super().__init__(params)
        self.data_class.N = params.N
        self.data_class.HARD = params.hard
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.BALANCED =params.balanced_search
        encoding_augmentation = random_symmetry_adj_matrix if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "vector_k_integers":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=self.k, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, nosep=params.nosep, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, nosep=params.nosep, pow2base=params.pow2base, encoding_function=encoding_augmentation)
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")


    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--N', type=int, default=30, help='Number of vertices in the K-cycle-free graph')
        parser.add_argument('--hard', type=bool_flag, default="true", help='whether only K-cycle-free graphs are accepted')
        parser.add_argument('--encoding_tokens', type=str, default="single_integer", help='single_integer/sequence_k_tokens/vector_k_integers/adjacency')
        parser.add_argument('--make_object_canonical', type=bool_flag, default="false", help="sort the graph node names based on its indegree")
        parser.add_argument('--augment_data_representation', type=bool_flag, default="false", help="augment the data representation with predefined function")
        parser.add_argument('--shuffle_elements', type=bool_flag, default="false", help="shuffle the elements of the adjacency matrix")
        parser.add_argument('--nosep', type=bool_flag, default="true", help='separator (for adjacency and double edge)')
        parser.add_argument('--pow2base', type=int, default=1, help='Number of adjacency entries to code together')
        parser.add_argument('--balanced_search', type=bool_flag, default="false", help="sort the graph node names based on its indegree")
        

class SquareEnvironment(CycleEnvironment):
    data_class = SquareDataPoint
    def __init__(self, params):
        super().__init__(params)


class TriangleEnvironment(CycleEnvironment):
    data_class = TriangleDataPoint
    def __init__(self, params):
        super().__init__(params)
