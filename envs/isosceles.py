from envs.environment import DataPoint, BaseEnvironment
import numpy as np
from numba import njit
from .tokenizers import SparseTokenizer, DenseTokenizer, EdgeTokenizer
from utils import bool_flag


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
def _greedy_add_jittered(matrix, candidates, N):
    max_points = N * N
    points_arr = np.empty((max_points, 2), dtype=np.int32)
    n_points = 0
    
    for x in range(N):
        for y in range(N):
            if matrix[x, y] == 1:
                points_arr[n_points, 0] = x
                points_arr[n_points, 1] = y
                n_points += 1
    
    for enc in candidates:
        x, y = enc // N, enc % N
        if matrix[x, y] == 1:
            continue
        
        if not _has_isosceles_conflict(points_arr, n_points, x, y):
            matrix[x, y] = 1
            points_arr[n_points, 0] = x
            points_arr[n_points, 1] = y
            n_points += 1


@njit(cache=True)
def _greedy_remove_jittered(matrix, triangles, N):
    num_triangles = len(triangles)
    if num_triangles == 0:
        return
    
    active = np.ones(num_triangles, dtype=np.bool_)
    
    point_count = np.zeros((N, N), dtype=np.int32)
    for t in range(num_triangles):
        point_count[triangles[t, 0], triangles[t, 1]] += 1
        point_count[triangles[t, 2], triangles[t, 3]] += 1
        point_count[triangles[t, 4], triangles[t, 5]] += 1
    
    num_active = num_triangles
    
    while num_active > 0:
        max_count = 0
        best_x, best_y = -1, -1
        for x in range(N):
            for y in range(N):
                if matrix[x, y] == 1 and point_count[x, y] > max_count:
                    max_count = point_count[x, y]
                    best_x, best_y = x, y
        
        if max_count == 0:
            break
        
        matrix[best_x, best_y] = 0
        
        for t in range(num_triangles):
            if not active[t]:
                continue
            
            contains = (
                (triangles[t, 0] == best_x and triangles[t, 1] == best_y) or
                (triangles[t, 2] == best_x and triangles[t, 3] == best_y) or
                (triangles[t, 4] == best_x and triangles[t, 5] == best_y)
            )
            
            if contains:
                active[t] = False
                num_active -= 1
                point_count[triangles[t, 0], triangles[t, 1]] -= 1
                point_count[triangles[t, 2], triangles[t, 3]] -= 1
                point_count[triangles[t, 4], triangles[t, 5]] -= 1


class NoIsoscelesDataPoint(DataPoint):
    N = 4
    HARD = True
    PENALTY = 6

    def __init__(self, init=False):
        super().__init__()
        self.matrix = np.zeros((self.N, self.N), dtype=np.int32)
        self.isosceles = np.empty((0, 6), dtype=np.int32)
        if init:
            self._add_points_greedily()
            self.calc_score()
            self.calc_features()

    def calc_score(self):
        if self.HARD and self.isosceles.size > 0:
            self.score = -1
            return
        self.score = self.matrix.sum().item() - self.PENALTY * self.isosceles.shape[0]

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
        _greedy_add_jittered(self.matrix, candidates, self.N)

    def _remove_points_greedily(self):
        if self.isosceles.size > 0:
            _greedy_remove_jittered(self.matrix, self.isosceles, self.N)
            self.isosceles = np.empty((0, 6), dtype=np.int32)

    def _isosceles_computation(self):
        points = np.argwhere(self.matrix == 1)
        points_arr = np.ascontiguousarray(points, dtype=np.int32)
        self.isosceles = _greedy_fill_jittered(points_arr, len(points_arr))

    def local_search(self):
        self._isosceles_computation()
        self._remove_points_greedily()
        self._add_points_greedily()
        self._isosceles_computation()
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


class NoIsoscelesEnvironment(BaseEnvironment):
    data_class = NoIsoscelesDataPoint
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
            self.tokenizer = DenseTokenizer(params.N, self.data_class,params.nosep, params.pow2base)
            self.symbols = [str(i) for i in range(2**params.pow2base)]
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
        parser.add_argument('--pow2base', type=int, default=1, help='Number of adjacency entries to code together')

