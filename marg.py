import reepy
from dataloader import Geolife
from tqdm import tqdm
from pathlib import Path
import filters
from reeb_utils import make_pickle_safe

import pickle

from paths import ensure_parent


def compute_multi_agent_reeb(sim, output_path, epsilon=1e-3, MAX_AGENT=None):
    normal_agents = list(set(range(sim.agent_count)) - set(sim.anomalous_agents))

    reeb = make_pickle_safe(
        reepy.SequentialReebGraph(epsilon=epsilon, store_trajectories=True)
    )

    for agent_id in tqdm(normal_agents):
        if MAX_AGENT and agent_id >= MAX_AGENT:
            break
        agent = sim.agent(agent_id)

        for traj in agent.samples():
            reeb.append_trajectory(traj)

    reeb.build()

    print("Filtering Reeb Graph")
    filters.make_finite_reeb(reeb)
    filters.make_complete_reeb(reeb)

    print("Writing MARG to disk")
    make_pickle_safe(reeb)
    ensure_parent(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(reeb, f)


if __name__ == "__main__":
    # save this reeb graph to the same location as the simulation
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for resolution in ("1e-3", "1e-4", "1e-5", "1e-6"):
        compute_multi_agent_reeb(
            Geolife(split_date=None),
            f"outputs/MARG_geolife_epsilon={resolution}.pkl",
            epsilon=float(resolution),
            MAX_AGENT=None,
        )
