import os
import sys

print(sys.path)

import pytest
import asyncio
from typing import List, Tuple
from bayesien_paths import BayesMap, Bob, Alice
from private_path_query_utils import Grid, Path, compile_private_path_query


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


@pytest.mark.asyncio
async def test_compilation():
    result = await compile_private_path_query(num_parties=3, grid_size=3, query_size=3)


@pytest.mark.asyncio
async def test_run_computation():
    bob_grid = [
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]
    # create Bobs
    bobs = [Bob(grid=Grid(bob_grid)) for _ in range(2)]
    # create Alice
    alice = Alice(start=(0, 0), goal=(2, 2), path_lengths=3, grid_size=3)
    final_result = await alice.run_computation(bobs=bobs)


if __name__ == "__main__":

    pytest.main()
