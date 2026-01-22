from envs.environment import DataPoint, BaseEnvironment
from .utils import canonical_form_2d_symmetric, random_symmetry_2d_symmetric
import numpy as np
from numba import njit
from .tokenizers import SparseTokenizer, DenseTokenizer
from utils import bool_flag


# best_results = {
#     4: 6,
#     5: 7,
#     6: 9,
#     7: 10,
#     8: 13,
#     9: 16,
#     10: 18,
#     11: 18,
#     12: 20,
#     13: 22,
#     14: 23,
#     15: 25,
#     16: 28,
#     17: 30,
#     18: 32,
#     21: 36,
#     23: 40,
#     25: 44,
#     27: 48,
#     32: 56,
# }


@njit(cache=True)
def _greedy_fill_jittered(points_arr, n_points):
    if n_points < 3:
        return np.empty((0, 6), dtype=np.int32)
    
    max_triangles = n_points * (n_points - 1) * (n_points - 2) // 2
    triangles = np.empty((max_triangles, 6), dtype=np.int32)
    idx = 0
    
    for i in range(n_points):
        ax, ay = points_arr[i, 0], points_arr[i, 1]
        
        n_others = n_points - 1
        if n_others < 2:
            continue
        
        distances = np.empty(n_others, dtype=np.int64)
        indices = np.empty(n_others, dtype=np.int32)
        
        k = 0
        for j in range(n_points):
            if j == i:
                continue
            bx, by = points_arr[j, 0], points_arr[j, 1]
            dx = ax - bx
            dy = ay - by
            distances[k] = dx * dx + dy * dy
            indices[k] = j
            k += 1
        
        order = np.argsort(distances)
        
        start = 0
        while start < n_others:
            d2 = distances[order[start]]
            end = start + 1
            while end < n_others and distances[order[end]] == d2:
                end += 1
            
            if end - start >= 2:
                for p in range(start, end):
                    for q in range(p + 1, end):
                        j1 = indices[order[p]]
                        j2 = indices[order[q]]
                        triangles[idx, 0] = ax
                        triangles[idx, 1] = ay
                        triangles[idx, 2] = points_arr[j1, 0]
                        triangles[idx, 3] = points_arr[j1, 1]
                        triangles[idx, 4] = points_arr[j2, 0]
                        triangles[idx, 5] = points_arr[j2, 1]
                        idx += 1
            
            start = end
    
    return triangles[:idx]


@njit(cache=True)
def _has_isosceles_conflict(points_arr, n_points, new_x, new_y):
    if n_points < 2:
        return False
    
    new_distances = np.empty(n_points, dtype=np.int64)
    for i in range(n_points):
        dx = new_x - points_arr[i, 0]
        dy = new_y - points_arr[i, 1]
        new_distances[i] = dx * dx + dy * dy
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if new_distances[i] == new_distances[j]:
                return True
    
    for i in range(n_points):
        d_new = new_distances[i]
        for j in range(n_points):
            if i == j:
                continue
            dx = points_arr[i, 0] - points_arr[j, 0]
            dy = points_arr[i, 1] - points_arr[j, 1]
            if dx * dx + dy * dy == d_new:
                return True
    
    return False


@njit(cache=True)
def _expand_to_real_matrix(matrix, matrix_real, N):
    for x in range(N):
        for y in range(N):
            if matrix[x, y] == 1:
                matrix_real[x, y] = 1
                matrix_real[2 * N - 1 - x, y] = 1
                matrix_real[x, 2 * N - 1 - y] = 1
                matrix_real[2 * N - 1 - x, 2 * N - 1 - y] = 1


@njit(cache=True)
def _greedy_add_symmetric(matrix, matrix_real, candidates, N):
    max_points = 4 * N * N
    points_arr = np.empty((max_points, 2), dtype=np.int32)
    n_points = 0
    
    for x in range(2 * N):
        for y in range(2 * N):
            if matrix_real[x, y] == 1:
                points_arr[n_points, 0] = x
                points_arr[n_points, 1] = y
                n_points += 1
    
    for enc in candidates:
        x, y = enc // N, enc % N
        if matrix[x, y] == 1:
            continue
        
        sym_points = [
            (x, y),
            (2 * N - 1 - x, y),
            (x, 2 * N - 1 - y),
            (2 * N - 1 - x, 2 * N - 1 - y)
        ]
        
        # Temporarily add the 3 symmetric points to points_arr 
        # no need to remove if it doesn't work because it will rewrite from the next point
        for i in range(3):
            sx, sy = sym_points[i + 1]
            points_arr[n_points + i, 0] = sx
            points_arr[n_points + i, 1] = sy
        
        if not _has_isosceles_conflict(points_arr, n_points + 3, x, y):
            matrix[x, y] = 1
            for sx, sy in sym_points:
                matrix_real[sx, sy] = 1
            points_arr[n_points + 3, 0] = x
            points_arr[n_points + 3, 1] = y
            n_points += 4


@njit(cache=True)
def _greedy_remove_symmetric(matrix, matrix_real, triangles, N, random_floats):
    num_triangles = len(triangles)
    if num_triangles == 0:
        return
    
    use_random = len(random_floats) > 0
    
    active = np.ones(num_triangles, dtype=np.uint8)
    
    point_count = np.zeros((2 * N, 2 * N), dtype=np.int32)
    for t in range(num_triangles):
        point_count[triangles[t, 0], triangles[t, 1]] += 1
        point_count[triangles[t, 2], triangles[t, 3]] += 1
        point_count[triangles[t, 4], triangles[t, 5]] += 1
    
    num_active = num_triangles
    
    if use_random:
        problematic_x = np.empty(N * N, dtype=np.int32)
        problematic_y = np.empty(N * N, dtype=np.int32)
    
    step_idx = 0
    
    while num_active > 0:
        if use_random:
            n_problematic = 0
            for x in range(N):
                for y in range(N):
                    if matrix[x, y] == 1:
                        total_count = (point_count[x, y] + 
                                       point_count[2 * N - 1 - x, y] + 
                                       point_count[x, 2 * N - 1 - y] + 
                                       point_count[2 * N - 1 - x, 2 * N - 1 - y])
                        if total_count > 0:
                            problematic_x[n_problematic] = x
                            problematic_y[n_problematic] = y
                            n_problematic += 1
            
            if n_problematic == 0:
                break
            
            idx = int(random_floats[step_idx] * n_problematic)
            step_idx += 1
            best_x = problematic_x[idx]
            best_y = problematic_y[idx]
        else:
            max_count = 0
            best_x, best_y = -1, -1
            for x in range(N):
                for y in range(N):
                    if matrix[x, y] == 1:
                        total_count = (point_count[x, y] + 
                                       point_count[2 * N - 1 - x, y] + 
                                       point_count[x, 2 * N - 1 - y] + 
                                       point_count[2 * N - 1 - x, 2 * N - 1 - y])
                        if total_count > max_count:
                            max_count = total_count
                            best_x, best_y = x, y
            
            if max_count == 0:
                break
        
        matrix[best_x, best_y] = 0
        
        sym_points = [
            (best_x, best_y),
            (2 * N - 1 - best_x, best_y),
            (best_x, 2 * N - 1 - best_y),
            (2 * N - 1 - best_x, 2 * N - 1 - best_y)
        ]
        
        for sx, sy in sym_points:
            matrix_real[sx, sy] = 0
        
        for t in range(num_triangles):
            if not active[t]:
                continue
            
            contains = False
            for sx, sy in sym_points:
                if ((triangles[t, 0] == sx and triangles[t, 1] == sy) or
                    (triangles[t, 2] == sx and triangles[t, 3] == sy) or
                    (triangles[t, 4] == sx and triangles[t, 5] == sy)):
                    contains = True
                    break
            
            if contains:
                active[t] = False
                num_active -= 1
                point_count[triangles[t, 0], triangles[t, 1]] -= 1
                point_count[triangles[t, 2], triangles[t, 3]] -= 1
                point_count[triangles[t, 4], triangles[t, 5]] -= 1


class NoIsoscelesSymmetricDataPoint(DataPoint):
    HARD = True
    PENALTY = 6
    MAKE_OBJECT_CANONICAL = False
    BALANCED = False

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.matrix = np.zeros((self.N, self.N), dtype=np.uint8)
        self.matrix_real = np.zeros((2 * self.N, 2 * self.N), dtype=np.uint8)
        self.isosceles = np.empty((0, 6), dtype=np.int32)
        if init:
            self._add_points_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.matrix = canonical_form_2d_symmetric(self.matrix)
                self._sync_matrix_real()
            self.calc_features()
            self.calc_score()

    @classmethod
    def _init_from_existing_data(cls, N, old_data, mutation):
        assert N == old_data.N + 1
        new_data = cls(N=N, init=False)
        # be careful here: we are adding more points on the border, not on the interior 
        new_data.matrix[1:old_data.N+1, 1:old_data.N+1] = old_data.matrix

        new_data.mutate_and_search(n=mutation)
        return new_data
        
    def _sync_matrix_real(self):
        self.matrix_real.fill(0)
        _expand_to_real_matrix(self.matrix, self.matrix_real, self.N)

    def calc_score(self):
        if self.HARD and self.isosceles.size > 0:
            self.score = -1
            return
        self.score = self.matrix_real.sum().item() - self.PENALTY * self.isosceles.shape[0]

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(self.N):
                w.append(self.matrix[i, j])
        self.features = ",".join(map(str, w))

    def _add_points_greedily(self):
        np.random.seed(None)
        candidates = np.arange(self.N * self.N, dtype=np.int32)
        np.random.shuffle(candidates)
        _greedy_add_symmetric(self.matrix, self.matrix_real, candidates, self.N)

    def _remove_points_greedily(self):
        if self.isosceles.size > 0:
            if self.BALANCED:
                random_floats = np.random.random(self.N * self.N).astype(np.float32)
            else:
                random_floats = np.empty(0, dtype=np.float32)
            _greedy_remove_symmetric(self.matrix, self.matrix_real, self.isosceles, self.N, random_floats)
            self.isosceles = np.empty((0, 6), dtype=np.int32)

    def _isosceles_computation(self):
        points = np.argwhere(self.matrix_real == 1)
        points_arr = np.ascontiguousarray(points, dtype=np.int32)
        self.isosceles = _greedy_fill_jittered(points_arr, len(points_arr))

    def mutate_and_search(self, n):
        if n > 0:
            np.random.seed(None)
        for _ in range(np.random.randint(n+1)):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            self.matrix[i, j] = 1 - self.matrix[i, j]
        self._sync_matrix_real()
        self.local_search()

    def local_search(self):
        self._isosceles_computation()
        self._remove_points_greedily()
        self._add_points_greedily()
        self._isosceles_computation()
        self.calc_score()
        if self.MAKE_OBJECT_CANONICAL:
            self.matrix = canonical_form_2d_symmetric(self.matrix)
            self._sync_matrix_real()
        self.calc_features()

    @classmethod
    def _update_class_params(cls, pars):
        cls.HARD, cls.MAKE_OBJECT_CANONICAL, cls.BALANCED = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.HARD, cls.MAKE_OBJECT_CANONICAL, cls.BALANCED)


class NoIsoscelesSymmetricEnvironment(BaseEnvironment):
    # this problem lives in N^2, so we can use k=2
    # this problem is not symmetric, so we can use is_adj_matrix_symmetric=False
    k = 2
    is_adj_matrix_symmetric = False
    data_class = NoIsoscelesSymmetricDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.HARD = params.hard
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.BALANCED = params.balanced_search
        encoding_augmentation = random_symmetry_2d_symmetric if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "vector_k_integers":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=self.k, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, nosep=params.nosep, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, nosep=params.nosep, pow2base=params.pow2base, encoding_augmentation=encoding_augmentation)
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--min_N', type=int, default=30, help='Min half grid size N. Total grid size is 2N')
        parser.add_argument('--max_N', type=int, default=30, help='Max half grid size N. Total grid size is 2N')
        parser.add_argument('--hard', type=bool_flag, default="true", help='whether only isosceles-free point sets are accepted')
        parser.add_argument('--encoding_tokens', type=str, default="single_integer", help='single_integer/sequence_k_tokens/vector_k_integers/adjacency')
        parser.add_argument('--make_object_canonical', type=bool_flag, default="false", help="sort the grid by symmetry")
        parser.add_argument('--augment_data_representation', type=bool_flag, default="false", help="augment the data representation with predefined function")
        parser.add_argument('--shuffle_elements', type=bool_flag, default="false", help="shuffle the elements of the adjacency matrix")
        parser.add_argument('--nosep', type=bool_flag, default="true", help='separator (for adjacency and double edge)')
        parser.add_argument('--pow2base', type=int, default=1, help='Number of adjacency entries to code together')
        parser.add_argument('--balanced_search', type=bool_flag, default="false", help='when removing points on grid, do not be too greedy')