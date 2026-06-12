"""
Uses the Markov Chain approach to generate synthetic data on a per-agent level

Input: a list of stop-point matrices with 4 columns (x, y, start time, stop time)
"""

from tqdm import tqdm
import numpy as np
import h5py

import baseline.transforms as transforms
from baseline import MarkovChain

import tracers
from paths import ensure_parent

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

        M1.append(agent_m1)
        M2.append(agent_m2)

        # Training
        sps_m1 = transforms.extract_stop_points(agent_m1, epsilon)

        mc = MarkovChain(sps_m1)

        try: 
            G.append(tracers.baseline_tracer(mc, agent_m1, M2[0].shape[0]))
        except:
            print("Trace failed. Agent may have insufficient trajectories")
            M1.pop()
            M2.pop()
            continue


    print(f"Total number of agents: {len(M1)}")

    ensure_parent(output_path)
    with h5py.File(output_path, "w") as f:
        for group, agents in (("M1", M1), ("M2", M2), ("G", G)):
            for k, agent in enumerate(agents):
                f.create_dataset(f"{group}_{k}", data=agent)


if __name__ == "__main__":
    from dataloader import Geolife

    generate_single_agent_data(Geolife(), "outputs/baseline_geolife.h5")
