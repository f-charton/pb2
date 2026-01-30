from .threerank import ThreeRankEnvironment
from .sidon import SidonSetEnvironment
from .cycle import SquareEnvironment, TriangleEnvironment, TriangleSquareEnvironment
from .threeonetwo import ThreeOneTwoEnvironment
from .isosceles import NoIsoscelesEnvironment
from .isosceles_symmetric import NoIsoscelesSymmetricEnvironment
from .sphere import SphereEnvironment
from .sumdiff import SumDiffEnvironment

ENVS = {
    'threerank': ThreeRankEnvironment,
    'sidon': SidonSetEnvironment,
    'square': SquareEnvironment,
    'triangle': TriangleEnvironment,
    'triangle_square': TriangleSquareEnvironment,
    'threeonetwo': ThreeOneTwoEnvironment,
    'isosceles': NoIsoscelesEnvironment,
    'isosceles_symmetric': NoIsoscelesSymmetricEnvironment,
    'sphere': SphereEnvironment,
    'sumdiff': SumDiffEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
