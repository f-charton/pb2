from .threerank import ThreeRankEnvironment
#from .sidon import SidonEnvironment
from .square import SquareEnvironment
from .triangle import TriangleEnvironment

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
