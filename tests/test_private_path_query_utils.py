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
    Graph,
    GraphPath,
    Vertex,
    Edge,
    EdgeState,
    BasePath,
    Environment,
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
    assert result_unsat.satisfied_clauses is None
    assert result_unsat.u_vector is None
    assert result_unsat.hazards_on_path is None

    result_sat: ComputationResult = parse_output(output_sat)
    assert result_sat.is_solved
    assert result_sat.information_gain == 0.5
    assert result_sat.satisfied_clauses is None
    assert result_sat.u_vector is None
    assert result_sat.hazards_on_path is None


def test_parse_output_invalid():
    output_invalid = "Some output...\nNo relevant info here.\nMore output..."
    with pytest.raises(ValueError):
        parse_output(output_invalid)


def test_parse_output_verifier_safe():
    """Test parsing verifier output when path is safe."""
    output_safe = "Some output...\ninformation_gain= 0.42\nPath is safe: 1\nHazards on path (count of matches): 0\nMore output..."
    result: ComputationResult = parse_output(output_safe, ProgramName.VERIFIER)
    assert result.is_solved, "Path should be marked as safe (solved)"
    assert result.information_gain == 0.42
    assert result.hazards_on_path == 0


def test_parse_output_verifier_unsafe():
    """Test parsing verifier output when path is unsafe."""
    output_unsafe = "Some output...\ninformation_gain= 0.11\nPath is safe: 0\nHazards on path (count of matches): 3\nMore output..."
    result: ComputationResult = parse_output(output_unsafe, ProgramName.VERIFIER)
    assert not result.is_solved, "Path should be marked as unsafe (not solved)"
    assert result.information_gain == 0.11
    assert result.hazards_on_path == 3


def test_parse_output_verifier_invalid():
    """Test parsing verifier output with missing key."""
    output_invalid = "Some output...\nNo relevant info here.\nMore output..."
    with pytest.raises(ValueError):
        parse_output(output_invalid, ProgramName.VERIFIER)


def test_parse_output_find_safe_path():
    output_sat = "Some output...\nis_solved = 1\nsatisfied clauses = 64\nMore output..."
    output_unsat = (
        "Some output...\nis_solved = 0\nsatisfied clauses = 60\nMore output..."
    )

    result_sat: ComputationResult = parse_output(output_sat, ProgramName.FIND_SAFE_PATH)
    assert result_sat.is_solved
    assert result_sat.information_gain == 0.0
    assert result_sat.satisfied_clauses == 64.0
    assert result_sat.u_vector is None
    assert result_sat.hazards_on_path is None

    result_unsat: ComputationResult = parse_output(
        output_unsat, ProgramName.FIND_SAFE_PATH
    )
    assert not result_unsat.is_solved
    assert result_unsat.information_gain == 0.0
    assert result_unsat.satisfied_clauses == 60.0
    assert result_unsat.u_vector is None
    assert result_unsat.hazards_on_path is None


def test_parse_output_find_safe_path_structured_with_u_vector():
    output = "\n".join(
        [
            "RESULT_TYPE=MatSatResult",
            "RESULT_IS_SOLVED=0",
            "RESULT_SATISFIED_CLAUSES=60",
            "RESULT_U[0]=1",
            "RESULT_U[2]=1",
            "RESULT_U[1]=0",
        ]
    )
    result: ComputationResult = parse_output(output, ProgramName.FIND_SAFE_PATH)
    assert not result.is_solved
    assert result.information_gain == 0.0
    assert result.satisfied_clauses == 60.0
    assert result.u_vector == [1, 0, 1]
    assert result.hazards_on_path is None


def test_parse_output_verifier_structured():
    output = "\n".join(
        [
            "Path is safe: 1",
            "Hazards on path (count of matches): 0",
            "RESULT_TYPE=VerifierResult",
            "RESULT_IS_SOLVED=1",
            "RESULT_INFORMATION_GAIN=0.33",
            "RESULT_HAZARDS_ON_PATH=0",
        ]
    )
    result: ComputationResult = parse_output(output, ProgramName.VERIFIER)
    assert result.is_solved
    assert result.information_gain == 0.33
    assert result.satisfied_clauses is None
    assert result.u_vector is None
    assert result.hazards_on_path == 0


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


def _environment_size(env: Environment) -> int:
    """Get square environment size for Grid/Graph inputs."""
    if isinstance(env, Grid):
        return len(env.grid)
    if isinstance(env, Graph):
        return len(env.adjacency_list)
    raise TypeError(f"Unsupported environment type: {type(env)}")


def _graph_from_edge_states(edge_states: List[List[int]]) -> Graph:
    """Build a Graph object from an NxN edge-state matrix."""
    vertices = [Vertex(id=i) for i in range(len(edge_states))]
    return Graph(vertices=vertices, edge_states=edge_states)


async def _run_sat_test_helper(
    environments: List[Environment],
    path: BasePath,
    iteration_no: int = 0,
    is_graph: bool = False,
) -> ComputationResult:
    """
    Helper function to run a SAT test computation.

    Returns:
        ComputationResult with is_solved and information_gain
    """
    num_parties = len(environments) + 1
    grid_size = _environment_size(environments[0])
    query_size = len(path.moves)
    base_port = 5001

    ok = await compile_private_path_query(
        num_parties, grid_size, query_size, iteration_no, is_graph=is_graph
    )
    assert ok, "SPDZ compile failed"

    inputs = [path] + environments

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
    environments: List[Environment],
    path: BasePath,
    is_graph: bool = False,
    iteration_no: int = 0,
) -> ComputationResult:
    """
    Helper function to run a verifier test computation.

    Returns:
        ComputationResult with is_solved (True = safe, False = unsafe) and information_gain
    """
    num_parties = len(environments) + 1
    grid_size = _environment_size(environments[0])
    # In both grid and graph modes, query_size is the number of "steps":
    # - Grid Path: number of (dx, dy) moves
    # - GraphPath: number of edges (u, v)
    query_size = len(path.moves)
    base_port = 5002  # Use different port to avoid conflicts

    ok = await compile_verifier(
        num_parties=num_parties,
        grid_size=grid_size,
        query_size=query_size,
        is_graph=is_graph,
        iteration_no=iteration_no,
    )
    assert ok, "SPDZ verifier compile failed"

    inputs = [path] + environments

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
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_unsafe():
    """Test verifier with an unsafe path (has hazards)."""
    # Create grids with hazards along the path
    # Path: (0,0) -> (1,0) -> (2,0)
    # So we need hazards at (0,0), (1,0), or (2,0)
    grid_1 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])  # Hazards in first column
    grid_2 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])  # Same pattern
    path_1 = Path(
        start=(0, 0), moves=[(1, 0), (1, 0)]
    )  # Goes through (0,0), (1,0), (2,0)
    result = await _run_verifier_test_helper([grid_1, grid_2], path_1)
    assert not result.is_solved, "Path should be unsafe (has hazards)"
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_graph_safe():
    """Test verifier with a safe graph path (no hazardous edges on path)."""
    # Graph: 0-1-2-3 (linear chain as vertices)
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e23 = Edge(vertex1=v2, vertex2=v3)

    # Path: 0 -> 1 -> 2 (edges e01 and e12)
    path = GraphPath(start=v0, moves=[e01, e12])

    # Graph edge-state matrix: 2=traversable, 1=blocked, 0=no edge.
    # Make edge (2,3) blocked, but keep (0,1) and (1,2) traversable.
    edge_states_1 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],  # edge 2->3 blocked
        [0, 0, 0, 0],
    ]
    edge_states_2 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]

    graph_1 = _graph_from_edge_states(edge_states_1)
    graph_2 = _graph_from_edge_states(edge_states_2)

    result = await _run_verifier_test_helper([graph_1, graph_2], path, is_graph=True)
    assert result.is_solved, "Graph path should be safe (avoids hazardous edge 2->3)"
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_graph_unsafe():
    """Test verifier with an unsafe graph path (uses a hazardous edge)."""
    # Graph: 0-1-2-3 (linear chain as vertices)
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e23 = Edge(vertex1=v2, vertex2=v3)

    # Path: 0 -> 1 -> 2 -> 3 (includes edge 2->3 which is hazardous)
    path = GraphPath(start=v0, moves=[e01, e12, e23])

    edge_states_1 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],  # edge 2->3 blocked
        [0, 0, 0, 0],
    ]
    edge_states_2 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]

    graph_1 = _graph_from_edge_states(edge_states_1)
    graph_2 = _graph_from_edge_states(edge_states_2)

    result = await _run_verifier_test_helper([graph_1, graph_2], path, is_graph=True)
    assert not result.is_solved, "Path should be unsafe (has hazards)"
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_graph_complex():
    """Complex graph verifier test with multiple branches and hazards."""
    # Graph vertices: 0,1,2,3,4,5
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)
    v4 = Vertex(id=4)
    v5 = Vertex(id=5)

    # Edges: 0-1-2-3 is a straight line, and 1-4-5 is an alternate branch
    e01 = Edge(vertex1=v0, vertex2=v1)
    e10 = Edge(vertex1=v1, vertex2=v0)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e21 = Edge(vertex1=v2, vertex2=v1)
    e23 = Edge(vertex1=v2, vertex2=v3)
    e32 = Edge(vertex1=v3, vertex2=v2)
    e14 = Edge(vertex1=v1, vertex2=v4)
    e41 = Edge(vertex1=v4, vertex2=v1)
    e45 = Edge(vertex1=v4, vertex2=v5)
    e54 = Edge(vertex1=v5, vertex2=v4)

    # Safe path: 0 -> 1 -> 4 -> 5 (avoids hazardous edges on the 1-2-3 chain)
    safe_path = GraphPath(start=v0, moves=[e01, e14, e45])

    # Unsafe path: 0 -> 1 -> 2 -> 3 (uses hazardous edges on the main chain)
    unsafe_path = GraphPath(start=v0, moves=[e01, e12, e23])

    # Graph edge-state matrix: 2=traversable, 1=blocked, 0=no edge.
    # Make safe-path edges traversable and main-chain hazardous edges blocked.
    size = 6
    base_rows = [[0] * size for _ in range(size)]
    # Copy base_rows for each grid; we'll set hazards explicitly.
    grid_data_1 = [row[:] for row in base_rows]
    grid_data_2 = [row[:] for row in base_rows]

    # Mark traversable edges used by the safe path 0->1->4->5
    for g in (grid_data_1, grid_data_2):
        g[0][1] = 2
        g[1][4] = 2
        g[4][5] = 2
        # Mark blocked edges on main chain used by unsafe path
        g[1][2] = 1
        g[2][3] = 1

    graph_1 = _graph_from_edge_states(grid_data_1)
    graph_2 = _graph_from_edge_states(grid_data_2)

    # Safe path should avoid all hazardous edges
    result_safe = await _run_verifier_test_helper(
        [graph_1, graph_2], safe_path, is_graph=True
    )
    assert result_safe.is_solved, "Complex graph safe_path should be reported safe"
    assert result_safe.information_gain is not None
    assert result_safe.information_gain > 0

    # Unsafe path should traverse hazardous edges and be reported unsafe
    result_unsafe = await _run_verifier_test_helper(
        [graph_1, graph_2], unsafe_path, is_graph=True
    )
    assert not result_unsafe.is_solved, "Complex graph unsafe_path should be unsafe"
    assert result_unsafe.information_gain is not None
    assert result_unsafe.information_gain > 0


@pytest.mark.asyncio
async def test_join_computation_verifier_graph_complex_all_safe():
    """Complex graph verifier test where all edges are safe."""
    # Graph vertices: 0,1,2,3,4,5
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)
    v4 = Vertex(id=4)
    v5 = Vertex(id=5)

    # Richly connected graph with multiple routes from 0 to 5
    e01 = Edge(vertex1=v0, vertex2=v1)
    e10 = Edge(vertex1=v1, vertex2=v0)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e21 = Edge(vertex1=v2, vertex2=v1)
    e23 = Edge(vertex1=v2, vertex2=v3)
    e32 = Edge(vertex1=v3, vertex2=v2)
    e14 = Edge(vertex1=v1, vertex2=v4)
    e41 = Edge(vertex1=v4, vertex2=v1)
    e45 = Edge(vertex1=v4, vertex2=v5)
    e54 = Edge(vertex1=v5, vertex2=v4)
    e25 = Edge(vertex1=v2, vertex2=v5)
    e52 = Edge(vertex1=v5, vertex2=v2)

    # Choose a longer path that we expect to be entirely safe:
    # 0 -> 1 -> 2 -> 3 -> 2 -> 5
    safe_complex_path = GraphPath(start=v0, moves=[e01, e12, e23, e32, e25])

    # Graph edge-state matrix: initialize no-edge (0) then mark traversable edges (2)
    size = 6
    grid_data_1 = [[0] * size for _ in range(size)]
    grid_data_2 = [[0] * size for _ in range(size)]
    for g in (grid_data_1, grid_data_2):
        g[0][1] = 2
        g[1][2] = 2
        g[2][3] = 2
        g[3][2] = 2
        g[2][5] = 2
    graph_1 = _graph_from_edge_states(grid_data_1)
    graph_2 = _graph_from_edge_states(grid_data_2)

    result = await _run_verifier_test_helper(
        [graph_1, graph_2], safe_complex_path, is_graph=True
    )
    assert (
        result.is_solved
    ), "Complex all-safe graph path should be reported safe by verifier"
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


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
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


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
    assert result.information_gain is not None
    assert (
        result.information_gain > 0
    ), "Information gain should be positive for verifier"


@pytest.mark.asyncio
async def test_join_computation_verifier_info_gain_safe_and_unsafe():
    """Run both solved and unsolved verifier paths and assert positive information gain."""
    safe_grid_1 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    safe_grid_2 = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    safe_path = Path(start=(0, 0), moves=[(1, 0), (1, 0)])
    safe_result = await _run_verifier_test_helper(
        [safe_grid_1, safe_grid_2], safe_path, iteration_no=0
    )
    assert safe_result.is_solved
    assert safe_result.information_gain is not None
    assert safe_result.information_gain > 0

    unsafe_grid_1 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    unsafe_grid_2 = Grid([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    unsafe_path = Path(start=(0, 0), moves=[(1, 0), (1, 0)])
    unsafe_result = await _run_verifier_test_helper(
        [unsafe_grid_1, unsafe_grid_2], unsafe_path, iteration_no=0
    )
    assert not unsafe_result.is_solved
    assert unsafe_result.information_gain is not None
    assert unsafe_result.information_gain > 0


def test_graph_str_simple():
    """Test Graph.__str__ with a simple 3-vertex graph."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)

    graph = Graph(vertices=[v0, v1, v2], edges=[e01, e12])
    result = str(graph)

    # Adjacency list: v0 connects to v1, v1 connects to v0 and v2, v2 connects to v1
    expected = "0 2 0 \n2 0 2 \n0 2 0"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_str_no_edges():
    """Test Graph.__str__ with vertices but no edges."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)

    graph = Graph(vertices=[v0, v1], edges=[])
    result = str(graph)

    # Should be 2x2 matrix of all zeros
    expected = "0 0 \n0 0"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_str_complete_graph():
    """Test Graph.__str__ with a complete 3-vertex graph (all vertices connected)."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e02 = Edge(vertex1=v0, vertex2=v2)
    e12 = Edge(vertex1=v1, vertex2=v2)

    graph = Graph(vertices=[v0, v1, v2], edges=[e01, e02, e12])
    result = str(graph)

    # Complete graph: all off-diagonal entries are 2 (traversable), diagonal entries are 0
    expected = "0 2 2 \n2 0 2 \n2 2 0"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_path_str_simple():
    """Test GraphPath.__str__ with a simple path."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)

    path = GraphPath(start=v0, moves=[e01, e12])
    result = str(path)

    # Should output: "0 1\n1 2"
    expected = "0 1\n1 2"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_path_str_single_edge():
    """Test GraphPath.__str__ with a path containing a single edge."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)

    e01 = Edge(vertex1=v0, vertex2=v1)

    path = GraphPath(start=v0, moves=[e01])
    result = str(path)

    # Should output: "0 1"
    expected = "0 1"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_path_str_long_path():
    """Test GraphPath.__str__ with a longer path."""
    vertices = [Vertex(id=i) for i in range(5)]

    edges = [Edge(vertex1=vertices[i], vertex2=vertices[i + 1]) for i in range(4)]

    path = GraphPath(start=vertices[0], moves=edges)
    result = str(path)

    # Should output: "0 1\n1 2\n2 3\n3 4"
    expected_lines = [f"{i} {i+1}" for i in range(4)]
    expected = "\n".join(expected_lines)

    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_graph_path_str_cycle():
    """Test GraphPath.__str__ with a path that forms a cycle."""
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e20 = Edge(vertex1=v2, vertex2=v0)

    path = GraphPath(start=v0, moves=[e01, e12, e20])
    result = str(path)

    # Should output: "0 1\n1 2\n2 0"
    expected = "0 1\n1 2\n2 0"
    assert result == expected, f"Expected '{expected}', got '{result}'"


@pytest.mark.asyncio
async def test_graph_path_sat():
    """Test graph path query with SAT result (safe path)."""
    # Create a simple graph with 4 vertices (0, 1, 2, 3)
    # Graph structure: 0-1-2-3 (linear chain)
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e23 = Edge(vertex1=v2, vertex2=v3)

    # Create a path: 0 -> 1 -> 2 (uses edges (0,1) and (1,2))
    path = GraphPath(start=v0, moves=[e01, e12])

    # In graph mode, grid[i][j] = 2 means traversable edge; 0/1 are non-traversable.
    # For 4 vertices, we need a 4x4 grid
    # Path uses edges (0,1) and (1,2), so make those traversable (2).
    edge_states_1 = [
        [0, 2, 0, 0],  # vertex 0: traversable edge to 1
        [0, 0, 2, 0],  # vertex 1: traversable edge to 2
        [0, 0, 0, 0],  # vertex 2
        [0, 0, 0, 0],  # vertex 3
    ]
    edge_states_2 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    graph_1 = _graph_from_edge_states(edge_states_1)
    graph_2 = _graph_from_edge_states(edge_states_2)

    result = await _run_sat_test_helper(
        [graph_1, graph_2], path, iteration_no=0, is_graph=True
    )
    assert result.is_solved, "Path should be SAT (safe, avoids dangerous vertex 3)"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_graph_path_unsat():
    """Test graph path query with UNSAT result (unsafe path through dangerous vertex)."""
    # Create a simple graph with 4 vertices (0, 1, 2, 3)
    # Graph structure: 0-1-2-3 (linear chain)
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v3 = Vertex(id=3)

    e01 = Edge(vertex1=v0, vertex2=v1)
    e12 = Edge(vertex1=v1, vertex2=v2)
    e23 = Edge(vertex1=v2, vertex2=v3)

    # Create a path: 0 -> 1 -> 2 -> 3 (uses edges (0,1), (1,2), and (2,3))
    path = GraphPath(start=v0, moves=[e01, e12, e23])

    # In graph mode, grid[i][j] = 2 means traversable edge; 0/1 are non-traversable.
    # For 4 vertices, we need a 4x4 grid
    # Path uses edges (0,1), (1,2), and (2,3)
    # Make edge (2,3) blocked (1) so the path is unsafe
    edge_states_1 = [
        [0, 2, 0, 0],  # vertex 0: traversable edge to 1
        [0, 0, 2, 0],  # vertex 1: traversable edge to 2
        [0, 0, 0, 1],  # vertex 2: edge to 3 exists but blocked
        [0, 0, 0, 0],  # vertex 3
    ]
    edge_states_2 = [
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],  # edge 2->3 blocked
        [0, 0, 0, 0],
    ]
    graph_1 = _graph_from_edge_states(edge_states_1)
    graph_2 = _graph_from_edge_states(edge_states_2)

    result = await _run_sat_test_helper(
        [graph_1, graph_2], path, iteration_no=0, is_graph=True
    )
    assert (
        not result.is_solved
    ), "Path should be UNSAT (unsafe, goes through dangerous vertex 3)"
    assert result.information_gain > 0, "Information gain should be positive"


@pytest.mark.asyncio
async def test_verifier_graph_edge_state_mix_safe_and_unsafe_pair():
    """Verifier pair test using mixed edge states (2 traversable, 1 blocked, 0 no-edge)."""
    # Graph nodes: 0,1,2,3,4
    # Safe path uses traversable edges: 0->1->4
    # Unsafe path uses one blocked edge: 0->1->2
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v4 = Vertex(id=4)

    safe_path = GraphPath(
        start=v0, moves=[Edge(vertex1=v0, vertex2=v1), Edge(vertex1=v1, vertex2=v4)]
    )
    unsafe_path = GraphPath(
        start=v0, moves=[Edge(vertex1=v0, vertex2=v1), Edge(vertex1=v1, vertex2=v2)]
    )

    size = 5
    grid_data_1 = [[int(EdgeState.NO_EDGE)] * size for _ in range(size)]
    grid_data_2 = [[int(EdgeState.NO_EDGE)] * size for _ in range(size)]

    # traversable edges
    for g in (grid_data_1, grid_data_2):
        g[0][1] = int(EdgeState.TRAVERSABLE)
        g[1][4] = int(EdgeState.TRAVERSABLE)
        # blocked edge
        g[1][2] = int(EdgeState.BLOCKED)
        # keep at least one explicit no-edge relation
        g[2][3] = int(EdgeState.NO_EDGE)

    graph_1 = _graph_from_edge_states(grid_data_1)
    graph_2 = _graph_from_edge_states(grid_data_2)

    result_safe = await _run_verifier_test_helper(
        [graph_1, graph_2], safe_path, is_graph=True
    )
    assert result_safe.is_solved, "Safe path should verify with mixed edge states"

    result_unsafe = await _run_verifier_test_helper(
        [graph_1, graph_2], unsafe_path, is_graph=True
    )
    assert not result_unsafe.is_solved, "Blocked-edge path should not verify"


@pytest.mark.asyncio
async def test_private_query_graph_edge_state_mix_sat_and_unsat_pair():
    """private_path_query SAT/UNSAT pair test using mixed edge states."""
    # Graph nodes: 0,1,2,3,4
    # SAT path uses only traversable edges: 0->1->4
    # UNSAT path uses blocked edge: 0->1->2
    v0 = Vertex(id=0)
    v1 = Vertex(id=1)
    v2 = Vertex(id=2)
    v4 = Vertex(id=4)

    sat_path = GraphPath(
        start=v0, moves=[Edge(vertex1=v0, vertex2=v1), Edge(vertex1=v1, vertex2=v4)]
    )
    unsat_path = GraphPath(
        start=v0, moves=[Edge(vertex1=v0, vertex2=v1), Edge(vertex1=v1, vertex2=v2)]
    )

    size = 5
    grid_data_1 = [[int(EdgeState.NO_EDGE)] * size for _ in range(size)]
    grid_data_2 = [[int(EdgeState.NO_EDGE)] * size for _ in range(size)]

    for g in (grid_data_1, grid_data_2):
        g[0][1] = int(EdgeState.TRAVERSABLE)
        g[1][4] = int(EdgeState.TRAVERSABLE)
        g[1][2] = int(EdgeState.BLOCKED)
        g[2][3] = int(EdgeState.NO_EDGE)

    graph_1 = _graph_from_edge_states(grid_data_1)
    graph_2 = _graph_from_edge_states(grid_data_2)

    result_sat = await _run_sat_test_helper(
        [graph_1, graph_2], sat_path, iteration_no=0, is_graph=True
    )
    assert result_sat.is_solved, "Traversable-edge path should be SAT"

    result_unsat = await _run_sat_test_helper(
        [graph_1, graph_2], unsat_path, iteration_no=0, is_graph=True
    )
    assert not result_unsat.is_solved, "Blocked-edge path should be UNSAT"


if __name__ == "__main__":
    pytest.main()
