# pytest tests/ -x
import pytest

try:
    import os
    from environs import Env

    env = Env()
    env.read_env(recurse=True)

except Exception as e:
    pass


def test_imports():
    import os
    from environs import Env


def test_finds_env():
    from environs import Env
    env = Env()
    env.read_env(recurse=True)


def test_paths():
    for path in ['WORKINGDIR', 'TRAINDIR', 'DATADIR', 'IMAGEDIR']:
        assert env.str(path) is not None, f"{path} is None"
        assert os.path.exists(env.str(path)), f"{env.str(path)} does not exist"