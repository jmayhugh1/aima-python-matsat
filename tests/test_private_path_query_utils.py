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
    compile_verifier,
    join_computation,
    Grid,
    Path,
    parse_output,
    ComputationResult,
    delete_persistence,
    ProgramName,
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


def test_parse_output_verifier_safe():
    """Test parsing verifier output when path is safe."""
    output_safe = "Some output...\nPath is safe: 1\nHazards on path (count of matches): 0\nMore output..."
    result: ComputationResult = parse_output(output_safe, ProgramName.VERIFIER)
    assert result.is_solved, "Path should be marked as safe (solved)"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


def test_parse_output_verifier_unsafe():
    """Test parsing verifier output when path is unsafe."""
    output_unsafe = "Some output...\nPath is safe: 0\nHazards on path (count of matches): 3\nMore output..."
    result: ComputationResult = parse_output(output_unsafe, ProgramName.VERIFIER)
    assert not result.is_solved, "Path should be marked as unsafe (not solved)"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


def test_parse_output_verifier_invalid():
    """Test parsing verifier output with missing key."""
    output_invalid = "Some output...\nNo relevant info here.\nMore output..."
    with pytest.raises(ValueError):
        parse_output(output_invalid, ProgramName.VERIFIER)


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


async def _run_verifier_test_helper(
    grids: List[Grid], path: Path
) -> ComputationResult:
    """
    Helper function to run a verifier test computation.

    Returns:
        ComputationResult with is_solved (True = safe, False = unsafe) and information_gain (0.0)
    """
    num_parties = len(grids) + 1
    grid_size = len(grids[0].grid)
    query_size = len(path.moves)
    base_port = 5002  # Use different port to avoid conflicts

    ok = await compile_verifier(num_parties, grid_size, query_size)
    assert ok, "SPDZ verifier compile failed"

    inputs = [path] + grids

    tasks = [
        asyncio.create_task(
            join_computation(
                id=i,
                num_parties=num_parties,
                input=inputs[i],
                port=base_port,
                program_name=ProgramName.VERIFIER,
            )
        )
        for i in range(num_parties)
    ]

    results = await asyncio.gather(*tasks)

    final_result = results[0]  # assuming party 0 returns the result
    return final_result


@pytest.mark.asyncio
async def test_compile_verifier():
    """Test that compile_verifier compiles successfully."""
    ok = await compile_verifier(num_parties=3, grid_size=3, query_size=2)
    assert ok, "Verifier compilation should succeed"


@pytest.mark.asyncio
async def test_join_computation_verifier_safe():
    """Test verifier with a safe path (no hazards)."""
    grid_1 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    grid_2 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    path_1 = Path(start=(0, 0), moves=[(1, 0), (1, 0)])
    result = await _run_verifier_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be safe (no hazards)"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_unsafe():
    """Test verifier with an unsafe path (has hazards)."""
    # Create grids with hazards along the path
    # Path: (0,0) -> (1,0) -> (2,0)
    # So we need hazards at (0,0), (1,0), or (2,0)
    grid_1 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])  # Hazards in first column
    grid_2 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])  # Same pattern
    path_1 = Path(start=(0, 0), moves=[(1, 0), (1, 0)])  # Goes through (0,0), (1,0), (2,0)
    result = await _run_verifier_test_helper([grid_1, grid_2], path_1)
    assert not result.is_solved, "Path should be unsafe (has hazards)"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_partial_hazard():
    """Test verifier with a path that has some hazards but not all."""
    # Path: (0,0) -> (0,1) -> (0,2)
    # Grid has hazards at (0,1) but not at (0,0) or (0,2)
    grid_1 = Grid([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    grid_2 = Grid([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    path_1 = Path(start=(0, 0), moves=[(0, 1), (0, 1)])
    result = await _run_verifier_test_helper([grid_1, grid_2], path_1)
    # Path should be unsafe because it hits a hazard at (0,1)
    assert not result.is_solved, "Path should be unsafe (hits hazard at (0,1))"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_5x5():
    """Test verifier with a larger 5x5 grid."""
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

    # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4) -> (3,4) -> (4,4)
    # This path avoids the hazards in grid_1 (row 1, cols 0,1,3) and grid_2 (row 3, cols 1,2,3)
    path_1 = Path(
        start=(0, 0),
        moves=[(0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0)],
    )

    result = await _run_verifier_test_helper([grid_1, grid_2], path_1)
    assert result.is_solved, "Path should be safe (avoids all hazards)"
    assert result.information_gain == 0.0, "Information gain should be 0.0 for verifier"


if __name__ == "__main__":
    pytest.main()
