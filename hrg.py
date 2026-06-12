import reepy
from tqdm import tqdm
from pathlib import Path
import pickle
import networkx as nx
import numpy as np
from sortedcontainers import SortedDict
import h5py
import copy
from collections import deque

import tracers
import filters
from paths import ensure_parent
from reeb_utils import make_pickle_safe


def __reweight_from_terminals_in_place(
    G, terminals, terminal_value=1.0, weight_key="weight"
):
    P = {u: 0.0 for u in G.nodes()}
    for t in terminals:
        P[t] = float(terminal_value)

    GR = G.reverse(copy=False)

    q = deque(terminals)
    seen = set(terminals)

    while q:
        v = q.popleft()

        # For each predecessor A of v in the original graph (successor in GR)
        for a in GR.neighbors(v):
            # Read old edge prob, write new weight back onto G in place
            old_p_ab = G[a][v][weight_key]
            new_w = old_p_ab * P[v]

            G[a][v][weight_key] = new_w

            # Update node probability as max over outgoing edges
            if new_w > P[a]:
                P[a] = new_w

            if a not in seen:
                seen.add(a)
                q.append(a)

    return P


def compute_hybrid_agent_data(
    sim, marg_path, output_path, alpha=1e-4, beta=0.9, MAX_AGENTS=None
):
    normal_agents = list(set(range(sim.agent_count)) - set(sim.anomalous_agents))

    if MAX_AGENTS:
        normal_agents = normal_agents[:MAX_AGENTS]

    # load in MARG
    marg = None
    with open(marg_path, "rb") as f:
        marg = pickle.load(f)
    make_pickle_safe(marg)

    epsilon = marg.epsilon

    marg_nodes = SortedDict()
    N = max(marg.nodes()) + 1
    L = marg.L

    for node, data in marg.nodes(data=True):
        marg_nodes.setdefault(data["time"], []).append((node, data))

    K = len(normal_agents)

    (
        M1,
        M2,
        G,
    ) = [], [], []

    for k, agent_id in enumerate(tqdm(normal_agents)):
        agent = sim.agent(agent_id)

        if "train" not in agent.data or "test" not in agent.data:
            print("[WARN] Skipping agent", agent_id)
            continue

        agent_reeb = make_pickle_safe(
            reepy.SequentialReebGraph(epsilon=epsilon, store_trajectories=True)
        )
        try:
            agent_reeb.append_trajectories(agent.samples())
        except:
            print("Trace failed. Agent may have insufficient trajectories")
            continue

        filters.make_finite_reeb(agent_reeb)
        filters.make_complete_reeb(agent_reeb)

        # rename all nodes in agent_reeb (prefix)
        rename_map = {n: n + N for n in agent_reeb.nodes()}
        nx.relabel_nodes(agent_reeb, rename_map, copy=False)

        # extract source and sink nodes of agent_reeb
        appear_nodes = {n for n, d in agent_reeb.nodes(data=True) if d["time"] == 0}
        disappear_nodes = {
            n for n, d in agent_reeb.nodes(data=True) if d["time"] == L - 1
        }

        # compute the corresponding nodes (twins) for agent/marg
        twins = []

        for node, data in agent_reeb.nodes(data=True):
            # we should ban twins for start and end points of trajectories
            if data["time"] in {0, L - 1}:
                continue

            marg_matches = marg_nodes.get(data["time"], [])

            nearest_node, nearest_data = min(
                marg_matches,
                key=lambda item: agent_reeb.dist(
                    np.array(data["centroid"]), np.array(item[1]["centroid"])
                ),
            )

            separation = agent_reeb.dist(
                np.array(nearest_data["centroid"]), np.array(data["centroid"])
            )
            if separation >= agent_reeb.epsilon:
                print(f"[WARN] Epsilon {separation} >= {agent_reeb.epsilon}")
                pass
            else:
                twins.append((node, nearest_node))

        # reindex agent trajectories -- we will add the agent trajectories to
        # the hybrid_reeb.trajectories manually
        for node, data in agent_reeb.nodes(data=True):
            agent_reeb.nodes[node]["trajs"] = tuple(
                np.array(data["trajs"]) + marg.trajc
            )

        # copy the MARG and add it to agent_reeb -- compose
        hybrid_reeb = copy.deepcopy(marg)
        hybrid_reeb.update(agent_reeb)
        assert (
            hybrid_reeb.number_of_nodes()
            == agent_reeb.number_of_nodes() + marg.number_of_nodes()
        )
        hybrid_reeb.trajc = marg.trajc + agent_reeb.trajc
        hybrid_reeb.trajectories = marg.trajectories + agent_reeb.trajectories
        assert len(hybrid_reeb.trajectories) == hybrid_reeb.trajc

        # draw edges between MARG and agent reeb with alpha/beta weights
        for agent_node, marg_node in twins:
            hybrid_reeb.add_edge(agent_node, marg_node, weight=alpha)
            hybrid_reeb.add_edge(marg_node, agent_node, weight=beta)

        # Prune unreachable from appear nodes (in_degree = 0 and in MARG)
        reachable = set()
        for node in appear_nodes:
            reachable |= nx.descendants(hybrid_reeb, node) | {node}

        hybrid_reeb.remove_nodes_from(
            [n for n in hybrid_reeb.nodes if n not in reachable]
        )

        # update the disappear nodes
        disappear_nodes = [n for n in disappear_nodes if n in hybrid_reeb]

        # Prune no path to disappear nodes
        reverse_hybrid_reeb = hybrid_reeb.reverse(copy=False)
        rreachable = set()
        for node in disappear_nodes:
            rreachable |= nx.descendants(reverse_hybrid_reeb, node) | {node}

        hybrid_reeb.remove_nodes_from(
            [n for n in hybrid_reeb.nodes if n not in rreachable]
        )

        # Reweight nodes in the MARG based on their relative costs
        # __reweight_from_terminals_in_place(hybrid_reeb, disappear_nodes)

        # Require probabilities to sum to 1
        for node in hybrid_reeb.nodes():
            out_edges = hybrid_reeb.out_edges(node, data=True)
            net_weight = sum(data["weight"] for _, _, data in out_edges)

            # only reweight if the weights don't add up to 1 already
            for _, v, data in out_edges:
                data["weight"] = data["weight"] / net_weight

        # generate trajectories
        agent_m1 = agent.samples("train")
        agent_m2 = agent.samples("test")

        assert agent_m1.shape[2] == 2, f"Unexpected shape: {agent_m1.shape}"

        M1.append(agent_m1)
        M2.append(agent_m2)

        try:
            G.append(tracers.random_tracer(hybrid_reeb, M2[0].shape[0]))
        except:
            print("Trace failed. Agent may have insufficient trajectories")
            M1.pop()
            M2.pop()
            continue

    ensure_parent(output_path)
    with h5py.File(output_path, "w") as f:
        for group, agents in (("M1", M1), ("M2", M2), ("G", G)):
            for k, agent in enumerate(agents):
                f.create_dataset(f"{group}_{k}", data=agent)


if __name__ == "__main__":
    """
    Generate Ablation HRGs
    """
    from dataloader import UrbanAnomalies, Geolife
    from multiprocessing import Pool

    def process_resolution(resolution, dataset_class, output_marg, output_hrg, alpha):
        """Process a single resolution for a given dataset"""
        compute_hybrid_agent_data(
            dataset_class(), 
            output_marg.format(resolution=resolution),
            output_hrg.format(resolution=resolution, alpha=alpha),
            alpha=alpha
        )
    
    resolutions = ["1e-3", "1e-4", "1e-5", "1e-6"]
    
    # Define tasks as tuples of (resolution, dataset_class, output_marg, output_hrg, alpha)
    tasks = []
    
    # Geolife tasks
    for resolution in resolutions:
        tasks.append((
            resolution,
            Geolife,
            "outputs/MARG_geolife_epsilon={resolution}.pkl",
            "outputs/hrg_geolife_epsilon={resolution}_alpha=1e-1.h5",
            1e-1
        ))
    
    # Get the number of CPU cores and use all of them
    num_cores = 1
    print(f"Using {num_cores} CPU cores to process {len(tasks)} tasks")
    
    # Run in parallel using all available cores
    with Pool(processes=num_cores) as pool:
        pool.starmap(process_resolution, tasks)
    
    print("All tasks completed!")
