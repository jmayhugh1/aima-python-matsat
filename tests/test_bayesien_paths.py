import os
import sys

print(sys.path)

import pytest
import asyncio
from typing import List, Tuple
from bayesien_paths import BayesMap


map_data_with_path = [
    [0, 0, 1],
    [1, 0, 1],
    [1, 0, 0],
]
map_data_without_path = [
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 0],
]
bayes_map_with_path: BayesMap = BayesMap(map=map_data_with_path)
bayes_map_wout_path: BayesMap = BayesMap(map=map_data_without_path)
bayes_map_empty: BayesMap = BayesMap(size=3, p_init=0.4)


def test_bayes_map_initialization():
    size = 3
    p_init = 0.2
    bayes_map = BayesMap(size=size, p_init=p_init)
    assert bayes_map.size == size
    assert bayes_map.p_init == p_init
    assert len(bayes_map.map) == size
    assert len(bayes_map.map[0]) == size


def test_check_viable_path():
    result = bayes_map_with_path.check_viable_path()
    assert result is True
    result = bayes_map_wout_path.check_viable_path()
    assert result is False


if __name__ == "__main__":

    pytest.main()
