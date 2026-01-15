from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger
import statistics
from collections import Counter, defaultdict
import numpy as np
from copy import deepcopy

logger=getLogger()


class DataPoint(ABC):
    def __init__(self):
        super().__init__()
        self.score = -1
        self.features = ""

    @abstractmethod
    def calc_score(self):
        pass

    @abstractmethod
    def calc_features(self):
        pass

    def local_search(self):
        return

    def redeem(self):
        return
    
    def mutate_and_search(self, n) -> None:
        self.local_search()

    def generate_and_score(self):
        self.calc_features()
        self.calc_score()
        return self if self.score >=0 else None

    @classmethod
    def _update_class_params(self,pars):
        return

    # TODO: reintroduce this
    # @classmethod
    # def _batch_generate_from_existing_data(cls, data, mutation, pars=None):
    #     out = []
    #     if pars is not None:
    #         cls._update_class_params(pars)
    #     for old_data in data:
    #         new_data = cls._init_from_existing_data(N=old_data.N+1, old_data=old_data, mutation=mutation)
    #         if new_data.score>=0:
    #             out.append(new_data)
    #     return out
        
    @classmethod
    def _batch_generate_and_score(cls, batch_size, min_N, max_N, pars=None):
        out = []
        if pars is not None:
            cls._update_class_params(pars)
        for _ in range(batch_size):
            # TODO: make sure we want to sample N uniformly
            N = np.random.randint(min_N, max_N + 1)
            d = cls(N=N, init=True)
            # print("HERE val",d.val)
            if d.score >=0:
                out.append(d)
        # print("HERE out",out)
        return out



class BaseEnvironment(object):
    data_class = None
    SPECIAL_SYMBOLS = ["SEP","EOS","PAD","BOS"]

    def __init__(self, params):
        return

    def read_data(self, input_file):
        data = []
        with open(input_file, "r") as file:
            for line in file:
                d = self.data_class.from_string(line)
                data.append(d)
        return data



def compute_stats(scores, label):
    num_bins = 200
    if len(scores) == 0:
        logger.info(f"[{label}] No valid examples")
        return None

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    top_1_percentile = statistics.quantiles(scores, n=100)[-1] if len(scores) >= 100 else max(scores)
    max_score = max(scores)
    
    logger.info(f"[{label}] Valid examples: {len(scores)}")
    logger.info(f"[{label}] Mean score: {mean}")
    logger.info(f"[{label}] Median score: {median}")
    logger.info(f"[{label}] Stdev score: {stdev}")
    logger.info(f"[{label}] Max score: {max_score}")
    logger.info(f"[{label}] Top 1 percentile score: {top_1_percentile}")

    logger.info(f"[{label}] Distribution of scores:")
    counts = Counter(sorted(scores))
    if len(counts) > num_bins:
        min_score, max_score = min(scores), max(scores)
        bin_width = (max_score - min_score) / num_bins
        bins = Counter()
        for score, count in counts.items():
            bin_idx = min(int((score - min_score) / bin_width), num_bins - 1)
            bin_start = min_score + bin_idx * bin_width
            bin_end = bin_start + bin_width
            bins[(bin_start, bin_end)] += count
        for (start, end), count in bins.items():
            logger.info(f"[{label}] Score [{start:.2f}, {end:.2f}): Count: {count}")
    else:
        for score, count in counts.items():
            logger.info(f"[{label}] Score {score}: Count: {count}")
    logger.info("--------------------------------")
    return {"mean": mean, "median": median, "top_1_percentile": top_1_percentile, "max": max_score}


def do_stats(n_invalid, data):
    """
    Compute and log statistics
    """
    logger.info(f"### Score distribution ###")
    if n_invalid >= 0:
        logger.info(f"Invalid examples: before local search: {n_invalid}, after: {len([d for d in data if d.score < 0])}")

    groups = defaultdict(list)
    for d in data:
        if d.score >= 0:
            groups[d.N].append(d.score)

    result = {}
    for n_value in sorted(groups.keys()):
        result[n_value] = compute_stats(scores=groups[n_value], label=f"N={n_value}")

    return result

def _do_score(d, always_search:bool = False, redeem:bool = False, offline_curriculum:bool = False, max_N:int = 0, mutation:int = 0, n_unique_searches:int = 1,pars=None):
    invalid = 0
    number_unique_searches = 1
    number_offline_unique_searches = 0
    if pars is not None:
        d._update_class_params(pars)
    d.calc_features()
    d.calc_score()
    if d.score < 0:
        invalid = 1
        if redeem:
            d.redeem()
    if always_search:
        res = []
        number_unique_searches = np.random.randint(1, n_unique_searches+1)
        for _ in range(number_unique_searches):
            temp_d = deepcopy(d)
            temp_d.mutate_and_search(mutation)
            res.append(temp_d)
        if offline_curriculum and d.N < max_N:
            number_offline_unique_searches = np.random.randint(1, n_unique_searches+1)
            for _ in range(number_offline_unique_searches):
                res.append(d._init_from_existing_data(N=d.N+1, old_data=d, mutation=mutation))
    else:
        res = [d]
    return (res, (number_unique_searches + number_offline_unique_searches) * invalid)

def do_score(data, args, executor=None):
    """
    Compute the score of a list of data.
    Can be parallelized with process_pool.
    Returns only valid items (score >= 0).
    """
    n_invalid = 0
    processed_data = []
    if not args.process_pool:
        for d in data:
            # warning, change the original list
            res, invalid = _do_score(d, args.always_search, args.redeem, args.offline_curriculum, args.max_N, args.mutation, args.n_unique_searches)
            n_invalid += invalid
            processed_data.extend(res)
    else:
        pars = data[0]._save_class_params()
        
        chunksize = max(1, len(data) // (args.num_workers * 32))
        
        if executor is not None:
            for d, invalid in executor.map(_do_score, data, repeat(args.always_search), repeat(args.redeem), repeat(args.offline_curriculum), repeat(args.max_N), repeat(args.mutation), repeat(args.n_unique_searches), repeat(pars), chunksize=chunksize):
                processed_data.extend(d)
                n_invalid += invalid
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                for d, invalid in ex.map(_do_score, data, repeat(args.always_search), repeat(args.redeem), repeat(args.offline_curriculum), repeat(args.max_N), repeat(args.mutation), repeat(args.n_unique_searches), repeat(pars), chunksize=chunksize):
                    processed_data.extend(d)
                    n_invalid += invalid

    valid_data = [d for d in processed_data if d.score >= 0]
    
    return valid_data, n_invalid, processed_data

