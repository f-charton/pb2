from envs.environment import DataPoint, BaseEnvironment
from .utils import canonical_form_3d, random_symmetry_3d
from numba import njit
import numpy as np
from utils import bool_flag
from .tokenizers import SparseTokenizer


# best_results = {
#     2: 4,  (validated from SAT solvers)
#     3: 8,  (validated from SAT solvers)
#     4: 11,  (validated from SAT solvers)
#     5: 14,  (validated from SAT solvers)
#     6: 18,  (bound from PB1)
#     7: 21,  (better bound from GDM)
#     8: 23,  (better bound from GDM)
#     9: 26,  (better bound from GDM)
#     10: 28,  (better bound from GDM)
#     11: 31,  (new bound from GDM)
#     12: 33,  (new bound from GDM)
# }

@njit(cache=True)
def _det2x2_elements(m00, m01, m10, m11):
    return m00 * m11 - m01 * m10


@njit(cache=True)
def _det3x3_from_matrix(matrix, r0, r1, r2, c0, c1, c2):
    m00 = matrix[r0, c0]
    m01 = matrix[r0, c1]
    m02 = matrix[r0, c2]
    m10 = matrix[r1, c0]
    m11 = matrix[r1, c1]
    m12 = matrix[r1, c2]
    m20 = matrix[r2, c0]
    m21 = matrix[r2, c1]
    m22 = matrix[r2, c2]

    return m00 * _det2x2_elements(m11, m12, m21, m22) - m01 * _det2x2_elements(m10, m12, m20, m22) + m02 * _det2x2_elements(m10, m11, m20, m21)


@njit(cache=True)
def _det4x4_from_matrix(matrix, r0, r1, r2, r3, c0, c1, c2, c3):
    d0 = _det3x3_from_matrix(matrix, r1, r2, r3, c1, c2, c3)
    d1 = _det3x3_from_matrix(matrix, r1, r2, r3, c0, c2, c3)
    d2 = _det3x3_from_matrix(matrix, r1, r2, r3, c0, c1, c3)
    d3 = _det3x3_from_matrix(matrix, r1, r2, r3, c0, c1, c2)

    return matrix[r0, c0] * d0 - matrix[r0, c1] * d1 + matrix[r0, c2] * d2 - matrix[r0, c3] * d3


@njit(cache=True)
def _det5x5_int(matrix):
    det = 0
    det += matrix[0, 0] * _det4x4_from_matrix(matrix, 1, 2, 3, 4, 1, 2, 3, 4)
    det -= matrix[0, 1] * _det4x4_from_matrix(matrix, 1, 2, 3, 4, 0, 2, 3, 4)
    det += matrix[0, 2] * _det4x4_from_matrix(matrix, 1, 2, 3, 4, 0, 1, 3, 4)
    det -= matrix[0, 3] * _det4x4_from_matrix(matrix, 1, 2, 3, 4, 0, 1, 2, 4)
    det += matrix[0, 4] * _det4x4_from_matrix(matrix, 1, 2, 3, 4, 0, 1, 2, 3)
    return det


@njit(cache=True)
def _are_five_points_cospherical(points):
    matrix = np.zeros((5, 5), dtype=np.int64)
    for i in range(5):
        p = points[i]
        matrix[i, 0] = p[0]
        matrix[i, 1] = p[1]
        matrix[i, 2] = p[2]
        matrix[i, 3] = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
        matrix[i, 4] = 1
    det = _det5x5_int(matrix)
    return det == 0


@njit(cache=True)
def _greedy_fill_jittered(points_arr, n_points):
    if n_points < 5:
        return np.empty((0, 15), dtype=np.int32)

    max_tuples = n_points * (n_points - 1) * (n_points - 2) * (n_points - 3) * (n_points - 4) // 120
    cospherical = np.empty((max_tuples, 15), dtype=np.int32)
    idx = 0

    points_to_check = np.zeros((5, 3), dtype=np.int64)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            for k in range(j + 1, n_points):
                for l in range(k + 1, n_points):
                    for m in range(l + 1, n_points):
                        points_to_check[0] = points_arr[i]
                        points_to_check[1] = points_arr[j]
                        points_to_check[2] = points_arr[k]
                        points_to_check[3] = points_arr[l]
                        points_to_check[4] = points_arr[m]

                        if _are_five_points_cospherical(points_to_check):
                            cospherical[idx, 0] = points_arr[i, 0]
                            cospherical[idx, 1] = points_arr[i, 1]
                            cospherical[idx, 2] = points_arr[i, 2]
                            cospherical[idx, 3] = points_arr[j, 0]
                            cospherical[idx, 4] = points_arr[j, 1]
                            cospherical[idx, 5] = points_arr[j, 2]
                            cospherical[idx, 6] = points_arr[k, 0]
                            cospherical[idx, 7] = points_arr[k, 1]
                            cospherical[idx, 8] = points_arr[k, 2]
                            cospherical[idx, 9] = points_arr[l, 0]
                            cospherical[idx, 10] = points_arr[l, 1]
                            cospherical[idx, 11] = points_arr[l, 2]
                            cospherical[idx, 12] = points_arr[m, 0]
                            cospherical[idx, 13] = points_arr[m, 1]
                            cospherical[idx, 14] = points_arr[m, 2]
                            idx += 1

    return cospherical[:idx]


@njit(cache=True)
def _has_cospherical_conflict(points_arr, n_points, new_x, new_y, new_z):
    if n_points < 4:
        return False

    points_to_check = np.zeros((5, 3), dtype=np.int64)
    points_to_check[4, 0] = new_x
    points_to_check[4, 1] = new_y
    points_to_check[4, 2] = new_z

    for i in range(n_points):
        for j in range(i + 1, n_points):
            for k in range(j + 1, n_points):
                for l in range(k + 1, n_points):
                    points_to_check[0] = points_arr[i]
                    points_to_check[1] = points_arr[j]
                    points_to_check[2] = points_arr[k]
                    points_to_check[3] = points_arr[l]

                    if _are_five_points_cospherical(points_to_check):
                        return True

    return False


@njit(cache=True)
def _greedy_add_jittered(matrix, candidates, N):
    max_points = N * N * N
    points_arr = np.empty((max_points, 3), dtype=np.int32)
    n_points = 0

    for x in range(N):
        for y in range(N):
            for z in range(N):
                if matrix[x, y, z] == 1:
                    points_arr[n_points, 0] = x
                    points_arr[n_points, 1] = y
                    points_arr[n_points, 2] = z
                    n_points += 1

    for enc in candidates:
        x = enc // (N * N)
        y = (enc // N) % N
        z = enc % N

        if matrix[x, y, z] == 1:
            continue

        if not _has_cospherical_conflict(points_arr, n_points, x, y, z):
            matrix[x, y, z] = 1
            points_arr[n_points, 0] = x
            points_arr[n_points, 1] = y
            points_arr[n_points, 2] = z
            n_points += 1


@njit(cache=True)
def _greedy_remove_jittered(matrix, cospherical, N, random_floats):
    num_tuples = len(cospherical)
    if num_tuples == 0:
        return

    use_random = len(random_floats) > 0

    active = np.ones(num_tuples, dtype=np.uint8)

    point_count = np.zeros((N, N, N), dtype=np.int32)
    for t in range(num_tuples):
        point_count[cospherical[t, 0], cospherical[t, 1], cospherical[t, 2]] += 1
        point_count[cospherical[t, 3], cospherical[t, 4], cospherical[t, 5]] += 1
        point_count[cospherical[t, 6], cospherical[t, 7], cospherical[t, 8]] += 1
        point_count[cospherical[t, 9], cospherical[t, 10], cospherical[t, 11]] += 1
        point_count[cospherical[t, 12], cospherical[t, 13], cospherical[t, 14]] += 1

    num_active = num_tuples

    if use_random:
        problematic_x = np.empty(N * N * N, dtype=np.int32)
        problematic_y = np.empty(N * N * N, dtype=np.int32)
        problematic_z = np.empty(N * N * N, dtype=np.int32)

    step_idx = 0

    while num_active > 0:
        if use_random:
            n_problematic = 0
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        if matrix[x, y, z] == 1 and point_count[x, y, z] > 0:
                            problematic_x[n_problematic] = x
                            problematic_y[n_problematic] = y
                            problematic_z[n_problematic] = z
                            n_problematic += 1

            if n_problematic == 0:
                break

            idx = int(random_floats[step_idx] * n_problematic)
            step_idx += 1
            best_x = problematic_x[idx]
            best_y = problematic_y[idx]
            best_z = problematic_z[idx]
        else:

            max_count = 0
            best_x, best_y, best_z = -1, -1, -1
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        if matrix[x, y, z] == 1 and point_count[x, y, z] > max_count:
                            max_count = point_count[x, y, z]
                            best_x, best_y, best_z = x, y, z

            if max_count == 0:
                break

        matrix[best_x, best_y, best_z] = 0

        for t in range(num_tuples):
            if not active[t]:
                continue

            contains = (
                (cospherical[t, 0] == best_x and cospherical[t, 1] == best_y and cospherical[t, 2] == best_z)
                or (cospherical[t, 3] == best_x and cospherical[t, 4] == best_y and cospherical[t, 5] == best_z)
                or (cospherical[t, 6] == best_x and cospherical[t, 7] == best_y and cospherical[t, 8] == best_z)
                or (cospherical[t, 9] == best_x and cospherical[t, 10] == best_y and cospherical[t, 11] == best_z)
                or (cospherical[t, 12] == best_x and cospherical[t, 13] == best_y and cospherical[t, 14] == best_z)
            )

            if contains:
                active[t] = False
                num_active -= 1
                point_count[cospherical[t, 0], cospherical[t, 1], cospherical[t, 2]] -= 1
                point_count[cospherical[t, 3], cospherical[t, 4], cospherical[t, 5]] -= 1
                point_count[cospherical[t, 6], cospherical[t, 7], cospherical[t, 8]] -= 1
                point_count[cospherical[t, 9], cospherical[t, 10], cospherical[t, 11]] -= 1
                point_count[cospherical[t, 12], cospherical[t, 13], cospherical[t, 14]] -= 1


class SphereDataPoint(DataPoint):
    N = 4
    HARD = True
    PENALTY = 10
    MAKE_OBJECT_CANONICAL = False
    BALANCED = False

    def __init__(self, init=False):
        super().__init__()
        self.matrix = np.zeros((self.N, self.N, self.N), dtype=np.uint8)
        self.cospherical = np.empty((0, 15), dtype=np.int32)
        if init:
            self._add_points_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.matrix = canonical_form_3d(self.matrix)
            self.calc_features()
            self.calc_score()

    def calc_score(self):
        if self.HARD and self.cospherical.size > 0:
            self.score = -1
            return
        self.score = self.matrix.sum().item() - self.PENALTY * self.cospherical.shape[0]

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    w.append(self.matrix[i, j, k])
        self.features = ",".join(map(str, w))

    def _add_points_greedily(self):
        np.random.seed(None)
        candidates = np.arange(self.N * self.N * self.N, dtype=np.int32)
        np.random.shuffle(candidates)
        _greedy_add_jittered(self.matrix, candidates, self.N)

    def _remove_points_greedily(self):
        if self.cospherical.size > 0:
            if self.BALANCED:
                random_floats = np.random.random(self.N * self.N * self.N).astype(np.float32)
            else:
                random_floats = np.empty(0, dtype=np.float32)
            _greedy_remove_jittered(self.matrix, self.cospherical, self.N, random_floats)
            self.cospherical = np.empty((0, 15), dtype=np.int32)

    def _cospherical_computation(self):
        points = np.argwhere(self.matrix == 1)
        points_arr = np.ascontiguousarray(points, dtype=np.int32)
        self.cospherical = _greedy_fill_jittered(points_arr, len(points_arr))

    def mutate_and_search(self, n):
        if n > 0:
            np.random.seed(None)
        for _ in range(np.random.randint(n+1)):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            k = np.random.randint(self.N)
            self.matrix[i, j, k] = 1
        self.local_search()

    def local_search(self):
        self._cospherical_computation()
        self._remove_points_greedily()
        self._add_points_greedily()
        self._cospherical_computation()
        self.calc_score()
        if self.MAKE_OBJECT_CANONICAL:
            self.matrix = canonical_form_3d(self.matrix)
        self.calc_features()

    @classmethod
    def _update_class_params(self, pars):
        self.N = pars[0]
        self.HARD = pars[1]
        self.MAKE_OBJECT_CANONICAL = pars[2]
        self.BALANCED = pars[3]

    @classmethod
    def _save_class_params(self):
        return (self.N, self.HARD, self.MAKE_OBJECT_CANONICAL, self.BALANCED)

    @classmethod
    def _batch_generate_and_score(cls, n, pars=None):
        return super()._batch_generate_and_score(n, pars)


class SphereEnvironment(BaseEnvironment):
    # this problem lives in N^3, so we can use k=3
    # this problem is not symmetric, so we can use is_adj_matrix_symmetric=False
    k = 3
    is_adj_matrix_symmetric = False
    data_class = SphereDataPoint
    def __init__(self, params):
        super().__init__(params)
        self.data_class.N = params.N
        self.data_class.HARD = params.hard
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.BALANCED = params.balanced_search
        encoding_augmentation = random_symmetry_3d if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "vector_k_integers":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=self.k, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=encoding_augmentation)
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizer(self.data_class, params.N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, nosep=params.nosep, encoding_augmentation=encoding_augmentation)
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--N", type=int, default=30, help="Number of vertices in the K-cycle-free graph")
        parser.add_argument("--hard", type=bool_flag, default="true", help="whether only K-cycle-free graphs are accepted")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer", help="single_integer/sequence_k_tokens/vector_k_integers")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="sort the matrix by symmetry")
        parser.add_argument('--augment_data_representation', type=bool_flag, default="false", help="augment the data representation with predefined function")
        parser.add_argument("--shuffle_elements", type=bool_flag, default="false", help="shuffle the elements of the adjacency matrix")
        parser.add_argument("--nosep", type=bool_flag, default="true", help="separator (for adjacency and double edge)")
        parser.add_argument('--balanced_search', type=bool_flag, default="false", help='when removing points on grid, do not be too greedy')
