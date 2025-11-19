"""Test script for MultiAgentWumpusKBMatSatSecure with secure multi-party computation.

This test demonstrates the secure multi-party computation approach where
each agent's private knowledge is kept private while still allowing
collaborative reasoning through MP-SPDZ.
"""

from simple_wumpus import (
    MultiAgentWumpusKBMatSatSecure,
    MultiAgentHybridWumpusAgent,
)
from agents4e import MultiWumpusEnvironment


def main():
    """Run the multi-agent secure Wumpus World test."""
    print("=" * 70)
    print("Multi-Agent Secure Wumpus World Test")
    print("Using MultiAgentWumpusKBMatSatSecure (MP-SPDZ)")
    print("=" * 70)
    print()

    # Setup multi-agent system with secure MatSat (MP-SPDZ)
    dimensions = 5
    agent_locations = [(1, 1), (3, 1), (1, 3)]
    shared_kb = MultiAgentWumpusKBMatSatSecure(
        dimrow=dimensions, agents_location=agent_locations
    )

    # Create agents with their correct starting locations
    agents = [
        MultiAgentHybridWumpusAgent(
            agent_id=i,
            shared_kb=shared_kb,
            dimensions=dimensions,
            start_location=agent_locations[i],
        )
        for i in range(len(agent_locations))
    ]

    # Create environment
    env = MultiWumpusEnvironment(
        agents_list=agents,
        agent_locations=agent_locations,
        width=dimensions,
        height=dimensions,
        show=True,
    )

    # Run simulation
    print("\nStarting simulation...")
    env.run(20)
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
