import os
import pytest
import asyncio
import tempfile
import shutil
import pathlib
from typing import List, Tuple
from unittest.mock import patch
from private_path_query_utils import (
    compile_private_path_query,
    join_computation,
    Grid,
    Path,
    parse_output,
    ComputationResult,
    delete_persistence,
)


def test_to_str_path():
    start = (0, 0)
    moves = [(1, 0), (0, 1), (1, 1)]
    path = Path(start=start, moves=moves)
    assert str(path) == "0\n0\n1\n0\n0\n1\n1\n1\n"


def test_to_str_grid():
    grid_data = [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    grid = Grid(grid=grid_data)
    assert str(grid) == "0 1 0\n1 0 1\n0 0 0"


def test_parse_output():
    output_sat = "Some output...\nis_solved= 1\ninformation_gain= 0.5\nMore output..."
    output_unsat = "Some output...\nis_solved= 0\ninformation_gain= 0.0\nMore output..."

    result_unsat: ComputationResult = parse_output(output_unsat)
    assert not result_unsat.is_solved
    assert result_unsat.information_gain == 0.0

    result_sat: ComputationResult = parse_output(output_sat)
    assert result_sat.is_solved
    assert result_sat.information_gain == 0.5


def test_parse_output_invalid():
    output_invalid = "Some output...\nNo relevant info here.\nMore output..."
    with pytest.raises(ValueError):
        parse_output(output_invalid)


def test_delete_persistence_folder_exists():
    """Test that delete_persistence deletes the Persistence folder when it exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock MP-SPDZ structure
        mock_spdz_root = pathlib.Path(tmpdir) / "MP-SPDZ"
        mock_spdz_root.mkdir()
        persistence_dir = mock_spdz_root / "Persistence"
        persistence_dir.mkdir()

        # Create a test file inside to verify deletion
        test_file = persistence_dir / "test_file.txt"
        test_file.write_text("test content")

        # Verify folder exists before deletion
        assert (
            persistence_dir.exists()
        ), "Persistence folder should exist before deletion"
        assert test_file.exists(), "Test file should exist before deletion"

        # Mock _SPDZ_ROOT to point to our temporary directory
        with patch("private_path_query_utils._SPDZ_ROOT", mock_spdz_root):
            delete_persistence()

        # Verify folder was deleted
        assert not persistence_dir.exists(), "Persistence folder should be deleted"
        assert not test_file.exists(), "Test file should be deleted"


def test_delete_persistence_folder_not_exists():
    """Test that delete_persistence handles missing folder gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock MP-SPDZ structure without Persistence folder
        mock_spdz_root = pathlib.Path(tmpdir) / "MP-SPDZ"
        mock_spdz_root.mkdir()
        persistence_dir = mock_spdz_root / "Persistence"

        # Verify folder doesn't exist
        assert not persistence_dir.exists(), "Persistence folder should not exist"

        # Mock _SPDZ_ROOT to point to our temporary directory
        with patch("private_path_query_utils._SPDZ_ROOT", mock_spdz_root):
            # Should not raise an error
            delete_persistence()

        # Verify folder still doesn't exist (no error occurred)
        assert not persistence_dir.exists(), "Persistence folder should still not exist"


async def _run_sat_test_helper(
    grids: List[Grid], path: Path, iteration_no: int = 0
) -> ComputationResult:
    """
    Helper function to run a SAT test computation.

    Returns:
        ComputationResult with is_solved and information_gain
    """
    num_parties = len(grids) + 1
    grid_size = len(grids[0].grid)
    query_size = len(path.moves)
    base_port = 5001

    ok = await compile_private_path_query(
        num_parties, grid_size, query_size, iteration_no
    )
    assert ok, "SPDZ compile failed"

    inputs = [path] + grids

    tasks = [
        asyncio.create_task(
            join_computation(
                id=i,
                num_parties=num_parties,
                input=inputs[i],
                port=base_port,  # SAME base port for all parties
            )
        )
        for i in range(num_parties)
    ]

    results = await asyncio.gather(*tasks)

    final_result = results[0]  # assuming party 0 returns the SAT decision
    return final_result


@pytest.mark.asyncio
async def test_join_computation_sat():
    grid_1 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    grid_2 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    path_1 = Path(start=(0, 0), moves=[(1, 0), (1, 0)])
    result = await _run_sat_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be sat"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_join_computation_unsat():
    grid_1 = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    grid_2 = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    path_1 = Path(start=(0, 0), moves=[(1, 0), (1, 0)])
    result = await _run_sat_test_helper([grid_1, grid_2], path_1)
    assert not result.is_solved, "Path should be unsat"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_join_computation_sat_5x5():
    grid_1 = Grid(
        [
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    grid_2 = Grid(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    # Visits:
    # (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,3)->(2,4)->(3,4)->(4,4)
    path_1 = Path(
        start=(0, 0),
        moves=[(0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0)],
    )

    result = await _run_sat_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be sat"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_join_computation_sat_6x6():
    grid_1 = Grid(
        [
            [0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    grid_2 = Grid(
        [
            [0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )

    # Visits:
    # (0,0)->(1,1)->(1,2)->(1,3)->(2,3)->(3,3)->(3,4)->(3,5)->(4,5)->(5,5)
    path_1 = Path(
        start=(0, 0),
        moves=[(1, 1), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0)],
    )

    result = await _run_sat_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be sat"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_join_computation_sat_7x7():
    grid_1 = Grid(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ]
    )
    grid_2 = Grid(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ]
    )

    # Visits:
    # (0,0)->(0,1)->(0,2)->(1,2)->(1,3)->(1,4)->(2,4)->(3,4)->(3,5)->(3,6)->(4,6)->(5,6)->(6,6)
    path_1 = Path(
        start=(0, 0),
        moves=[
            (0, 1),
            (0, 1),
            (1, 0),
            (0, 1),
            (0, 1),
            (1, 0),
            (1, 0),
            (0, 1),
            (0, 1),
            (1, 0),
            (1, 0),
            (1, 0),
        ],
    )

    result = await _run_sat_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be sat"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_join_computation_sat_2_iterations():
    """
    Test with 6x6 grid:
    - First iteration (iteration_no=0): unsat path through dangerous cells
    - Second iteration (iteration_no=1): sat path avoiding dangerous cells
    """
    # Create grids with dangerous cells in the middle-right area
    # Dangerous cells (1) are in columns 3-5, rows 1-4
    grid_1 = Grid(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    grid_2 = Grid(
        [
            [0, 0, 0, 1, 1, 1],  # Same pattern as grid_1
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    # First iteration: unsat path that goes through dangerous cells
    # Path: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4) -> (0,5)
    # This goes through dangerous cells at (0,3), (0,4), (0,5)
    path_unsat = Path(
        start=(0, 0),
        moves=[
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ],  # Moves right through dangerous area
    )

    # Second iteration: sat path that avoids dangerous cells
    # Path: (0,0) -> (1,0) -> (2,0) -> (3,0) -> (4,0) -> (5,0)
    # This stays in the safe left columns (0-2)
    path_sat = Path(
        start=(0, 0),
        moves=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)],  # Moves down through safe area
    )

    # First iteration: unsat path
    result_unsat = await _run_sat_test_helper(
        [grid_1, grid_2], path_unsat, iteration_no=0
    )
    assert not result_unsat.is_solved, "First iteration should be unsat"
    assert result_unsat.information_gain > 0, "Information gain should be positive"

    # Second iteration: sat path
    result_sat = await _run_sat_test_helper([grid_1, grid_2], path_sat, iteration_no=1)
    assert result_sat.is_solved, "Second iteration should be sat"
    assert result_sat.information_gain > 0, "Information gain should be positive"


if __name__ == "__main__":
    pytest.main()
