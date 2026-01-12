from envs.environment import DataPoint, BaseEnvironment
import numpy as np
from numba import njit
from .tokenizers import SparseTokenizer, DenseTokenizer
from utils import bool_flag


@njit(cache=True)
def _is_312(ax, ay, bx, by, cx, cy):
    return (
        (ay < by < cy and cx < ax < bx)
        or (ay < cy < by and bx < ax < cx)
        or (by < ay < cy and cx < bx < ax)
        or (by < cy < ay and ax < bx < cx)
        or (cy < ay < by and bx < cx < ax)
        or (cy < by < ay and ax < cx < bx)
    )


@njit(cache=True)
def _gray_bit_to_flip(i):
    n1 = (i - 1) ^ ((i - 1) >> 1)
    n2 = i ^ (i >> 1)
    d = n1 ^ n2
    j = 0
    temp = d & -d
    while temp > 1:
        temp >>= 1
        j += 1
    s = (n1 & d) == 0
    return j, s


@njit(cache=True)
def _ryser_permanent(matrix, N):
    mat = matrix.astype(np.float64)
    v = mat.sum(axis=1) / 2.0
    p = 1.0
    for val in v:
        p *= val
    
    sign_positive = True
    for i in range(1, 2 ** (N - 1)):
        col, subtract = _gray_bit_to_flip(i)
        if subtract:
            v -= mat[:, col]
        else:
            v += mat[:, col]
        
        prod = 1.0
        for val in v:
            prod *= val
        sign_positive = not sign_positive
        if sign_positive:
            p += prod
        else:
            p -= prod
    
    return p * 2.0


@njit(cache=True)
def _has_conflict(x_by_y, count_by_y, new_x, new_y, N):
    for y1 in range(N):
        if y1 != new_y and count_by_y[y1] > 0:
            for y2 in range(y1 + 1, N):
                if y2 != new_y and count_by_y[y2] > 0:
                    for i in range(count_by_y[y1]):
                        for j in range(count_by_y[y2]):
                            if _is_312(x_by_y[y1, i], y1, x_by_y[y2, j], y2, new_x, new_y):
                                return True
    return False


@njit(cache=True)
def _greedy_fill_jittered(matrix, candidates, N):
    # x_by_y[y, i] = x-coordinate of the i-th point with y-coordinate y
    x_by_y = np.empty((N, N), dtype=np.int32)
    count_by_y = np.zeros(N, dtype=np.int32)
    
    for x in range(N):
        for y in range(N):
            if matrix[x, y] == 1:
                x_by_y[y, count_by_y[y]] = x
                count_by_y[y] += 1
    
    for enc in candidates:
        x, y = enc // N, enc % N
        if matrix[x, y] == 1:
            continue
        
        if not _has_conflict(x_by_y, count_by_y, x, y, N):
            matrix[x, y] = 1
            x_by_y[y, count_by_y[y]] = x
            count_by_y[y] += 1


@njit(cache=True)
def _greedy_remove_jittered(matrix, cycles, N):
    num_cycles = len(cycles)
    active = np.ones(num_cycles, dtype=np.bool)
    
    point_count = np.zeros((N, N), dtype=np.int32)    
    for c in range(num_cycles):
        point_count[cycles[c, 0], cycles[c, 1]] += 1
        point_count[cycles[c, 2], cycles[c, 3]] += 1
        point_count[cycles[c, 4], cycles[c, 5]] += 1
    
    num_active = num_cycles
    
    while num_active > 0:
        max_count = 0
        best_x, best_y = -1, -1
        for x in range(N):
            for y in range(N):
                if point_count[x, y] > max_count:
                    max_count = point_count[x, y]
                    best_x, best_y = x, y
        
        if max_count == 0:
            break
        
        matrix[best_x, best_y] = 0
        
        for c in range(num_cycles):
            if not active[c]:
                continue
            
            contains = False
            if cycles[c, 0] == best_x and cycles[c, 1] == best_y:
                contains = True
            elif cycles[c, 2] == best_x and cycles[c, 3] == best_y:
                contains = True
            elif cycles[c, 4] == best_x and cycles[c, 5] == best_y:
                contains = True
            
            if contains:
                active[c] = False
                num_active -= 1
                point_count[cycles[c, 0], cycles[c, 1]] -= 1
                point_count[cycles[c, 2], cycles[c, 3]] -= 1
                point_count[cycles[c, 4], cycles[c, 5]] -= 1

    return matrix


@njit(cache=True)
def _cycles_computation_jittered(points_arr, n_points):
    # We can do either this or firstly count the cycles and then allocate the array
    # For our case, this is faster.
    max_cycles = (n_points * (n_points - 1) * (n_points - 2)) // 6
    cycles = np.empty((max_cycles, 6), dtype=np.int32)
    idx = 0
    
    for i in range(n_points):
        ax, ay = points_arr[i, 0], points_arr[i, 1]
        for j in range(i + 1, n_points):
            bx, by = points_arr[j, 0], points_arr[j, 1]
            for k in range(j + 1, n_points):
                cx, cy = points_arr[k, 0], points_arr[k, 1]
                if _is_312(ax, ay, bx, by, cx, cy):
                    cycles[idx, 0] = ax
                    cycles[idx, 1] = ay
                    cycles[idx, 2] = bx
                    cycles[idx, 3] = by
                    cycles[idx, 4] = cx
                    cycles[idx, 5] = cy
                    idx += 1
    
    return cycles[:idx]


class ThreeOneTwoDataPoint(DataPoint):
    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.matrix = np.eye(self.N, dtype=np.bool_)
        self.cycles = np.empty((0, 6), dtype=np.int32)
        if init:
            self._add_edges_greedily()
            self.calc_score()
            self.calc_features()

    def calc_score(self):
        self.score = _ryser_permanent(self.matrix, self.N)

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(self.N):
                w.append(self.matrix[i, j])
        self.features = ",".join(map(str, w))

    def _add_edges_greedily(self):
        np.random.seed(None)
        candidates = np.arange(self.N * self.N, dtype=np.int32)
        np.random.shuffle(candidates)
        _greedy_fill_jittered(self.matrix, candidates, self.N)

    def _remove_edges_greedily(self):
        if len(self.cycles) > 0:
            _greedy_remove_jittered(self.matrix, self.cycles, self.N)
            self.cycles = np.empty((0, 6), dtype=np.int32)

    def _cycles_computation(self):
        points = np.argwhere(self.matrix == 1)
        points_arr = np.ascontiguousarray(points, dtype=np.int32)
        self.cycles = _cycles_computation_jittered(points_arr, len(points_arr))

    def mutate_and_search(self, n):
        if n > 0:
            np.random.seed(None)
        for _ in range(np.random.randint(n+1)):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            self.matrix[i, j] = 1 - self.matrix[i, j]
        self.local_search()

    def local_search(self):
        self._cycles_computation()
        self._remove_edges_greedily()
        self._add_edges_greedily()
        self._cycles_computation()
        self.calc_score()
        self.calc_features()

    @classmethod
    def _update_class_params(self,pars):
        pass

    @classmethod
    def _save_class_params(self):
        return ()


class ThreeOneTwoEnvironment(BaseEnvironment):
    # this problem lives in N^2, so we can use k=2
    # this problem is not symmetric, so we can use is_adj_matrix_symmetric=False
    k = 2
    is_adj_matrix_symmetric = False
    data_class = ThreeOneTwoDataPoint
    def __init__(self, params):
        super().__init__(params)
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements)
        elif params.encoding_tokens == "vector_k_integers":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=self.k, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements)
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, nosep=params.nosep)
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, params.nosep, params.pow2base)
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")


    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--min_N', type=int, default=30, help='Min number of vertices in the 3-1-2 graph')
        parser.add_argument('--max_N', type=int, default=30, help='Max number of vertices in the 3-1-2 graph')
        parser.add_argument('--encoding_tokens', type=str, default="single_integer", help='single_integer/sequence_k_tokens/vector_k_integers/adjacency')
        parser.add_argument('--shuffle_elements', type=bool_flag, default="false", help="shuffle the elements of the adjacency matrix")
        parser.add_argument('--nosep', type=bool_flag, default="true", help='separator (for adjacency and double edge)')
        parser.add_argument('--pow2base', type=int, default=1, help='Number of adjacency entries to code together')
