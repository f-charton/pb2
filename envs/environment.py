from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger
import statistics
from collections import Counter
import numpy as np

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
    
    def generate_and_score(self):
        self.calc_features()
        self.calc_score()
        return self if self.score >=0 else None

    @classmethod
    def _update_class_params(self,pars):
        return

    @classmethod
    def _batch_generate_and_score(cls,n, pars=None):
        out = []
        if pars is not None:
            cls._update_class_params(pars)
        for _ in range(n):
            d = cls(init=True)
            if d.score >=0:
                out.append(d)
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
    



def do_stats(n_invalid, data):
    """
    Compute and log statistics
    """
    scores = [d.score for d in data if d.score >= 0]
    num_bins = 200
    logger.info(f"### Score distribution ###")
    if n_invalid >= 0:
        # Evaluation during training
        logger.info(f"Invalid examples: before local search: {n_invalid}, after: {len(data) - len(scores)}")
    if len(scores) > 0:
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        stdev = statistics.stdev(scores)
        top_1_percentile = statistics.quantiles(scores, n=100)[-1]
        max_score = max(scores)
        logger.info(f"Valid examples {len(scores)}")
        logger.info(f"Mean score: {mean}")
        logger.info(f"Median score: {median}")
        logger.info(f"stdev score: {stdev}")
        logger.info(f"Max score: {max_score}")
        logger.info(f"Top 1 percentile score: {top_1_percentile}")
        logger.info("distribution of scores")

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
                logger.info(f"Score [{start:.2f}, {end:.2f}): Count: {count}")
        else:
            for score, count in counts.items():
                logger.info(f"Score {score}: Count: {count}")
        return {"mean": mean, "median": median, "top_1_percentile": top_1_percentile, "max": max_score}
    return None

def _do_score(d, always_search:bool = False, redeem:bool = False, pars=None):
    invalid = 0
    if pars is not None:
        d._update_class_params(pars)
    d.calc_features()
    d.calc_score()
    if d.score < 0:
        invalid = 1
        if redeem:
            d.redeem()
    if always_search:
        d.local_search()
    return (d,invalid)

def do_score(data, process_pool: bool = False, num_workers :int = 20, always_search:bool = False, redeem:bool = False, executor=None):
    """
    Compute the score of a list of data.
    Can be parallelized with process_pool.
    Returns only valid items (score >= 0).
    """
    n_invalid = 0
    if not process_pool:
        for d in data:
            # warning, change the original list
            d,invalid = _do_score(d,always_search, redeem)
            n_invalid += invalid
        processed_data = data
    else:
        pars = data[0]._save_class_params()
        
        chunksize = max(1, len(data) // (num_workers * 32))
        processed_data = []
        
        if executor is not None:
            for d, invalid in executor.map(_do_score, data, repeat(always_search), repeat(redeem), repeat(pars), chunksize=chunksize):
                processed_data.append(d)
                n_invalid += invalid
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for d, invalid in ex.map(_do_score, data, repeat(always_search), repeat(redeem), repeat(pars), chunksize=chunksize):
                    processed_data.append(d)
                    n_invalid += invalid

    valid_data = [d for d in processed_data if d.score >= 0]
    
    return valid_data, n_invalid, processed_data

