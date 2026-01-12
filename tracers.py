"""
Tracers -- given a Reeb Graph and a number of trajectories to generate,
each tracer will generate a tensor of new trajectories
"""

from matplotlib.typing import RGBAColourType
import numpy as np
from reepy.SequentialReebGraph import SequentialReebGraph

from baseline import MarkovChain


def random_tracer(reeb_graph, N):
    assert reeb_graph.store, "Storing trajectories required for trace functions"
    L, D = reeb_graph.L, reeb_graph.D

    trajs = np.empty((N, L, D))

    for n in range(N):
        start_nodes = [(n, d) for n, d in reeb_graph.nodes(data=True) if d["time"] == 0]
        init_pointc = sum([len(node[1]["trajs"]) for node in start_nodes])
        start_probabilities = [
            len(node[1]["trajs"]) / init_pointc for node in start_nodes
        ]

        start_index = np.random.choice(len(start_nodes), p=start_probabilities)

        cursor = start_nodes[start_index][0]
        cursor_trajs = set(reeb_graph.nodes[cursor]["trajs"])
        cursor_time = reeb_graph.nodes[cursor]["time"]

        out_edges = list(reeb_graph.out_edges(cursor, data="weight"))

        while len(out_edges) > 0:
            edge_probabilities = [edge[2] for edge in out_edges]

            next_edge_idx = np.random.choice(len(out_edges), p=edge_probabilities)
            next_edge = out_edges[next_edge_idx]

            # find the common trajectories in this range
            next_cursor = next_edge[1]
            next_cursor_time = reeb_graph.nodes[next_cursor]["time"]
            next_cursor_trajs = set(reeb_graph.nodes[next_cursor]["trajs"])
            common_trajs = cursor_trajs & next_cursor_trajs

            if next_cursor_time == cursor_time:
                # Do not trace any trajectories, just update internal state
                cursor = next_cursor
                cursor_trajs = next_cursor_trajs
                out_edges = list(reeb_graph.out_edges(cursor, data="weight"))
                continue

            if len(common_trajs) == 0:
                print(cursor_trajs)
                print(next_cursor_trajs)
                print(next_cursor_time)

            trace_traj = np.random.choice(list(common_trajs))

            trajs[n, cursor_time:next_cursor_time] = reeb_graph.trajectories[
                trace_traj
            ][cursor_time:next_cursor_time]

            cursor = next_cursor
            cursor_trajs = set(reeb_graph.nodes[cursor]["trajs"])
            cursor_time = reeb_graph.nodes[cursor]["time"]
            out_edges = list(reeb_graph.out_edges(cursor, data="weight"))

        # last node is always at the disappear event
        trajs[n, -1] = reeb_graph.nodes[cursor]["centroid"]

    return trajs


def path_tracer(reeb_graph: SequentialReebGraph, path):
    assert reeb_graph.store, "Storing trajectories required for trace functions"
    traj = np.empty((0, reeb_graph.D))

    cursor = path[0]
    cursor_trajs = set(reeb_graph.nodes[cursor]["trajs"])
    cursor_time = reeb_graph.nodes[cursor]["time"]

    for next_cursor in path[1:]:
        next_cursor_time = reeb_graph.nodes[next_cursor]["time"]
        next_cursor_trajs = set(reeb_graph.nodes[next_cursor]["trajs"])
        common_trajs = cursor_trajs & next_cursor_trajs

        assert next_cursor_time != cursor_time

        trace_traj = np.random.choice(list(common_trajs))

        traj = np.vstack(
            (traj, reeb_graph.trajectories[trace_traj][cursor_time:next_cursor_time])
        )

        cursor = next_cursor
        cursor_trajs = next_cursor_trajs
        cursor_time = next_cursor_time

    traj = np.vstack((traj, reeb_graph.nodes[cursor]["centroid"]))
    return traj

def baseline_tracer(markov, reference, N):
    """
    Generate N trajectories using Markov chain and reference trajectories.
    
    Args:
        markov: MarkovChain object with states, transitions, departures, routes
        reference: (M1, L, 2) reference trajectories
        N: number of trajectories to generate
        
    Returns:
        (N, L, 2) array of generated trajectories
    """
    L = reference.shape[1]
    trajectories = np.zeros((N, L, 2))
    
    for n in range(N):
        traj = []
        state = 0
        time = 0
        
        while time < L:
            dep_probs = markov.departures[state].copy()
            dep_probs[:time] = 0
            
            if dep_probs.sum() == 0:
                remaining = L - time
                traj.append(np.tile(markov.states[state], (remaining, 1)))
                break
            
            dep_probs /= dep_probs.sum()
            next_time = np.random.choice(len(dep_probs), p=dep_probs)
            
            if next_time <= time:
                remaining = L - time
                traj.append(np.tile(markov.states[state], (remaining, 1)))
                break
            
            stationary_len = next_time - time
            traj.append(np.tile(markov.states[state], (stationary_len, 1)))
            
            trans_probs = markov.transitions[state]
            if trans_probs.sum() == 0:
                time = next_time
                continue
            
            trans_probs = trans_probs / trans_probs.sum()
            next_state = np.random.choice(len(trans_probs), p=trans_probs)
            
            route_key = (state, next_state)
            if route_key in markov.routes:
                traj_idx, start, end = markov.routes[route_key]
                trip = reference[traj_idx, start:end+1]
                traj.append(trip)
                time = next_time + len(trip)
            else:
                time = next_time
            
            state = next_state
        
        full_traj = np.vstack(traj)
        trajectories[n] = full_traj[:L]
    
    return trajectories
