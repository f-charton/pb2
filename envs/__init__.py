from .threerank import ThreeRankEnvironment

ENVS = {
    'threerank': ThreeRankEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
