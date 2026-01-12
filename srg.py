"""
Generates Single Agent Reeb Graphs for all normal agents and their corresponding
trajectories. Saves three tensors to disk: M1 normal, M2 normal and G normal.
"""

import reepy
from tqdm import tqdm
import numpy as np
import h5py

import tracers
import filters


def generate_single_agent_data(sim, output_path, MAX_AGENTS=None, epsilon=1e-4):
    normal_agents = set(range(sim.agent_count)) - set(sim.anomalous_agents)

    if MAX_AGENTS:
        normal_agents = list(normal_agents)[:MAX_AGENTS]

    K = len(normal_agents)

    (
        M1,
        M2,
        G,
    ) = [], [], []

    for k, agent_id in enumerate(tqdm(normal_agents)):
        agent = sim.agent(agent_id)

        if "train" not in agent.data or "test" not in agent.data:
            print("Skipping this agent...")
            continue

        agent_m1 = agent.samples("train")
        agent_m2 = agent.samples("test")

        assert agent_m1.shape[2] == 2, f"Unexpected shape: {agent_m1.shape}"

        agent_reeb = reepy.SequentialReebGraph(epsilon=epsilon, store_trajectories=True)
        try:
            agent_reeb.append_trajectories(agent_m1)
        except:
            print("Trace failed. Agent may have insufficient trajectories")
            continue

        filters.make_finite_reeb(agent_reeb)
        filters.make_complete_reeb(agent_reeb)

        M1.append(agent_m1)
        M2.append(agent_m2)

        try:
            G.append(tracers.random_tracer(agent_reeb, M2[0].shape[0]))
        except:
            print("Trace failed. Agent may have insufficient trajectories")
            M1.pop()
            M2.pop()
            continue

    with h5py.File(output_path, "w") as f:
        for group, agents in (("M1", M1), ("M2", M2), ("G", G)):
            for k, agent in enumerate(agents):
                f.create_dataset(f"{group}_{k}", data=agent)


if __name__ == "__main__":
    from dataloader import Geolife, UrbanAnomalies

    # generate_single_agent_data(Geolife(), "outputs/srg_geolife.h5")
    generate_single_agent_data(UrbanAnomalies(), "outputs/srg_ua.h5")
