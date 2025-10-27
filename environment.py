from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger
import statistics

logger=getLogger()


class DataPoint(ABC):
    def __init__(self, params):
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
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        stdev = statistics.stdev(scores)
        max_score = max(scores)
        logger.info(f"Valid examples {len(scores)}")
        logger.info(f"Mean score: {mean}")
        logger.info(f"Median score: {median}")
        logger.info(f"stdev score: {stdev}")
        logger.info(f"Max score: {max_score}")
    return

def _do_score(d, always_search:bool = False):
    invalid = 0
    d.calc_features()
    d.calc_score()
    if d.score < 0:
        invalid = 1
        d.local_search()
        # print(d.score)
        assert d.score >= 0
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
            d.calc_features()
            d.calc_score()
            if d.score < 0:
                n_invalid += 1
                d.local_search()
            elif always_search:
                d.local_search()
        processed_data = data
    else:
        chunksize = max(1, len(data) // (num_workers * 32))
        processed_data = []
        n_invalid = 0
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for d, invalid in ex.map(_do_score, data, repeat(always_search), chunksize=chunksize):
                assert d.score >= 0 # debug
                processed_data.append(d)
                n_invalid += invalid

    do_stats(n_invalid, processed_data)
    return [d for d in processed_data if d.score >= 0]

