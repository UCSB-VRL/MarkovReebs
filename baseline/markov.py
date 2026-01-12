import numpy as np
from scipy.spatial.distance import cdist

class MarkovChain:
    def __init__(self, stop_matrices, epsilon=1e-4, bins=12):
        """
        Cluster stop locations and compute transition and departure statistics.
        
        Args:
            stop_matrices: List of N arrays, each (4, M) with rows [x, y, start_idx, end_idx]
            epsilon: spatial threshold for clustering stop locations
            bins: number of bins for departure time histogram
        """
        all_locations = np.concatenate([m[:2].T for m in stop_matrices if m.shape[1] > 0], axis=0)
        
        if len(all_locations) == 0:
            self.states = np.empty((0, 2))
            self.transitions = np.empty((0, 0))
            self.departures = np.empty((0, bins))
            self.stops = []
            return
        
        states = []
        assigned = np.zeros(len(all_locations), dtype=bool)
        
        for i in range(len(all_locations)):
            if assigned[i]:
                continue
            dists = np.sqrt(np.sum((all_locations - all_locations[i])**2, axis=1))
            cluster = dists <= epsilon
            states.append(all_locations[cluster].mean(axis=0))
            assigned[cluster] = True
        
        self.states = np.array(states)
        K = len(self.states)
        
        self.stops = []
        for m in stop_matrices:
            if m.shape[1] == 0:
                self.stops.append(np.empty((3, 0)))
                continue
            locs = m[:2].T
            dists = cdist(locs, self.states)
            indices = np.argmin(dists, axis=1)
            self.stops.append(np.vstack([indices, m[2:3].astype(int), m[3:4].astype(int)]))
        
        max_time = max((m[2].max() if m.shape[1] > 0 else 0 for m in self.stops), default=0)
        bin_edges = np.linspace(0, max_time + 1, bins + 1)
        self.departures = np.zeros((K, bins))
        
        for m in self.stops:
            if m.shape[1] == 0:
                continue
            for j in range(m.shape[1]):
                state_idx = int(m[0, j])
                dep_time = m[2, j]
                bin_idx = np.searchsorted(bin_edges[1:], dep_time)
                bin_idx = min(bin_idx, bins - 1)
                self.departures[state_idx, bin_idx] += 1
        
        self.transitions = np.zeros((K, K))
        self.routes = {}
        
        for n, m in enumerate(self.stops):
            if m.shape[1] < 2:
                continue
            for j in range(m.shape[1] - 1):
                from_state = int(m[0, j])
                to_state = int(m[0, j + 1])
                
                if self.transitions[from_state, to_state] == 0:
                    start = int(m[2, j])
                    end = int(m[1, j + 1])
                    self.routes[(from_state, to_state)] = (n, start, end)
                
                self.transitions[from_state, to_state] += 1
        
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        self.transitions = np.divide(self.transitions, row_sums, where=row_sums > 0, 
                                     out=np.zeros_like(self.transitions))
