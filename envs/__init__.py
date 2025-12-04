from .threerank import ThreeRankEnvironment
from .sidon import SidonSetEnvironment
from .cycle import SquareEnvironment, TriangleEnvironment
from .threeonetwo import ThreeOneTwoEnvironment
from .isosceles import NoIsoscelesEnvironment

ENVS = {
    'threerank': ThreeRankEnvironment,
    'sidon': SidonSetEnvironment,
    'square': SquareEnvironment,
    'triangle': TriangleEnvironment,
    'threeonetwo': ThreeOneTwoEnvironment,
    'isosceles': NoIsoscelesEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
