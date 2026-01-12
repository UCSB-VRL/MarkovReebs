"""
Postprocessing utilities
"""

import numpy as np
import networkx as nx


def normalize(reeb):
    for node in reeb.nodes():
        edges = list(reeb.out_edges(node))
        if edges:
            total = sum(reeb.edges[e]["weight"] for e in edges)
            if total > 0:
                for e in edges:
                    reeb.edges[e]["weight"] /= total


def make_finite_reeb(reeb):
    # Remove nodes with non-finite centroid components
    reeb.remove_nodes_from(
        [n for n in reeb.nodes() if not np.all(np.isfinite(reeb.nodes[n]["centroid"]))]
    )

    normalize(reeb)


def make_complete_reeb(reeb):
    # Find all nodes with max time
    max_time = max(reeb.nodes[n]["time"] for n in reeb.nodes())
    max_nodes = [n for n in reeb.nodes() if reeb.nodes[n]["time"] == max_time]

    # Keep only nodes with path to any max_node
    reachable = set(max_nodes)
    for node in max_nodes:
        reachable |= nx.ancestors(reeb, node)
    reeb.remove_nodes_from([n for n in list(reeb.nodes()) if n not in reachable])

    normalize(reeb)
