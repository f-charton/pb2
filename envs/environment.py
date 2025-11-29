from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger
import statistics
from collections import Counter

logger=getLogger()


class DataPoint(ABC):
    def __init__(self):
        super().__init__()
        self.score = -1
        self.features = []

    @abstractmethod
    def calc_score(self):
        pass

    @abstractmethod
    def calc_features(self):
        pass

    def local_search(self):
        return
    
    def generate_and_score(self):
        self.calc_features()
        self.calc_scores()
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
            d = cls()
            if d.score >=0:
                out.append(d)
        return out 


    
class BaseEnvironment(object):
    data_class = None

    def __init__(self, params):
        self.symbols = []
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
    logger.info(f"### Score distribution ###")
    if n_invalid >= 0:
        # Evaluation during training
        logger.info(f"Invalid examples: before local search: {n_invalid}, after: {len(data) - len(scores)}")
    if len(scores) > 0:
        counts = Counter(sorted(scores))
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        stdev = statistics.stdev(scores)
        max_score = max(scores)
        logger.info(f"Valid examples {len(scores)}")
        logger.info(f"Mean score: {mean}")
        logger.info(f"Median score: {median}")
        logger.info(f"stdev score: {stdev}")
        logger.info(f"Max score: {max_score}")
        logger.info("distribution of scores")
        for score, count in counts.items():
            logger.info(f"Score {score}: Count: {count}")
    return

def _do_score(d, always_search:bool = False,pars=None):
    invalid = 0
    if pars is not None:
        d._update_class_params(pars)
    d.calc_features()
    d.calc_score()
    if d.score < 0:
        invalid = 1
        d.local_search()
    elif always_search:
        d.local_search()
    return (d,invalid)

def do_score(data, process_pool: bool = False, num_workers :int = 20, always_search:bool = False):
    """
    Compute the score of a list of data.
    Can be parallelized with process_pool.
    Returns only valid items (score >= 0).
    """
    n_invalid = 0
    if not process_pool:
        for d in data:
            # warning, change the original list
            d,invalid = _do_score(d,always_search)
            n_invalid += invalid
        processed_data = data
    else:
        pars = data[0]._save_class_params()
        
        chunksize = max(1, len(data) // (num_workers * 32))
        processed_data = []
        n_invalid = 0
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for d, invalid in ex.map(_do_score, data, repeat(always_search), repeat(pars), chunksize=chunksize):
                # Line below not true for all problems
                # assert d.score >= 0 # debug
                processed_data.append(d)
                n_invalid += invalid

    do_stats(n_invalid, processed_data)
    return [d for d in processed_data if d.score >= 0]

