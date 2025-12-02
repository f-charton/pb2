from .threerank import ThreeRankEnvironment
#from .sidon import SidonEnvironment
from .cycle import SquareEnvironment, TriangleEnvironment

ENVS = {
    'threerank': ThreeRankEnvironment,
#    'sidon': SidonEnvironment,
    'square': SquareEnvironment,
    'triangle': TriangleEnvironment
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
