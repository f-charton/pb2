from envs.environment import DataPoint, BaseEnvironment
import math
import numpy as np
from .tokenizers import SparseTokenizer
from utils import bool_flag


class SumDiffDataPoint(DataPoint):
    MIN_SET_SIZE = 3
    ITERATIONS = 10
    PROBLEM = None

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.matrix = np.zeros(2 * N + 1, dtype=np.uint8)
        self.set_size = 0
        self.sumset_size = 0
        self.diffset_size = 0

        if init:
            self._init_random_set()
            self.calc_features()
            self.calc_score()

    @classmethod
    def _init_from_existing_data(cls, N, old_data, mutation):
        assert N >= old_data.N

        new_data = cls(N=N, init=False)
        offset = N - old_data.N
        for i in range(2 * old_data.N + 1):
            new_data.matrix[i + offset] = old_data.matrix[i]
        new_data.mutate_and_search(n=mutation)
        return new_data

    def _init_random_set(self):
        np.random.seed(None)
        set_size = np.random.randint(self.MIN_SET_SIZE, 2 * self.N + 1)
        indices = np.random.choice(2 * self.N + 1, size=set_size, replace=False)
        self.matrix[indices] = 1

    def _get_set_A(self):
        return [i - self.N for i in range(2 * self.N + 1) if self.matrix[i] == 1]

    def _compute_sumset(self, A_list):
        sumset = set()
        n = len(A_list)
        for i in range(n):
            for j in range(i, n):
                sumset.add(A_list[i] + A_list[j])
        return sumset

    def _compute_diffset(self, A_list):
        diffset = set()
        for a in A_list:
            for b in A_list:
                diffset.add(a - b)
        return diffset

    def calc_score(self):
        A_list = self._get_set_A()
        set_size = len(A_list)
        
        if set_size < self.MIN_SET_SIZE:
            self.score = 0
            return

        sumset_size = len(self._compute_sumset(A_list))
        diffset_size = len(self._compute_diffset(A_list))

        if self.PROBLEM == "6.42":
            ratio_sum = sumset_size / set_size
            ratio_diff = diffset_size / set_size
            if ratio_sum <= 1 or ratio_diff <= 1:
                self.score = 0
                return
            self.score = math.log(ratio_sum) / math.log(ratio_diff)
        elif self.PROBLEM == "6.43":
            if diffset_size <= 1 or sumset_size <= 1:
                self.score = 0
                return
            self.score = math.log(diffset_size) / math.log(sumset_size)

    def calc_features(self):
        elements = self._get_set_A()
        self.features = ",".join(map(str, sorted(elements)))

    def mutate_and_search(self, n):
        if n > 0:
            np.random.seed(None)
        for _ in range(np.random.randint(1, n + 1)):
            i = np.random.randint(2 * self.N + 1)
            self.matrix[i] = 1 - self.matrix[i]
        self.local_search()

    def local_search(self):
        np.random.seed(None)

        self.calc_score()
        current_score = self.score
        current_size = int(np.sum(self.matrix))
        for _ in range(self.ITERATIONS):
            candidates = np.arange(2 * self.N + 1, dtype=np.int32)
            np.random.shuffle(candidates)

            for c in candidates:
                self.matrix[c] = 1 - self.matrix[c]
                new_size = current_size + (1 if self.matrix[c] == 1 else -1)

                if new_size >= self.MIN_SET_SIZE:
                    self.calc_score()
                    if self.score > current_score:
                        current_score = self.score
                        current_size = new_size
                    else:
                        self.matrix[c] = 1 - self.matrix[c]
                else:
                    self.matrix[c] = 1 - self.matrix[c]

            self.calc_features()
            self.calc_score()

    @classmethod
    def _update_class_params(cls, pars):
        cls.MIN_SET_SIZE, cls.ITERATIONS, cls.PROBLEM = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.MIN_SET_SIZE, cls.ITERATIONS, cls.PROBLEM)


class SumDiffEnvironment(BaseEnvironment):
    # this problem lives in N^1, so we can use k=1
    data_class = SumDiffDataPoint
    k = 1
    is_adj_matrix_symmetric = False

    def __init__(self, params):
        super().__init__(params)
        self.data_class.PROBLEM = params.problem
        self.data_class.MIN_SET_SIZE = params.min_set_size
        self.data_class.ITERATIONS = params.iterations
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizer(self.data_class, params.min_N, params.max_N, self.k, self.is_adj_matrix_symmetric, self.SPECIAL_SYMBOLS, token_embeddings=1, encoding=params.encoding_tokens, shuffle_elements=params.shuffle_elements, encoding_augmentation=None)
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument('--problem', type=str, default="6.42", choices=["6.42", "6.43"], help='Problem type: 6.42 or 6.43')
        parser.add_argument('--min_N', type=int, default=500, help='Total vector size is 2N+1, [-N, N]')
        parser.add_argument('--max_N', type=int, default=500, help='Total vector size is 2N+1, [-N, N]')
        parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for local search')
        parser.add_argument('--min_set_size', type=int, default=5, help='Minimum size of set A')
        parser.add_argument('--encoding_tokens', type=str, default="single_integer", help='single_integer/sequence_k_tokens/vector_k_integers/adjacency')
        parser.add_argument('--shuffle_elements', type=bool_flag, default="false", help="shuffle the elements of the adjacency matrix")

