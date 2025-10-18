from abc import ABC, abstractmethod
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

def do_score(data):
    n_invalid = 0
    for d in data:
        d.calc_features()
        d.calc_score()
        if d.score < 0:
            n_invalid += 1
            d.local_search()
    
    do_stats(n_invalid,data)
    return [d for d in data if d.score >= 0]

