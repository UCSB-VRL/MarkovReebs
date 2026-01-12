import numpy as np

def detect_stops(agents, radius=1e-4, min_duration=90):
    """Detect stop points in agent trajectories.
    
    Args:
        agents: List of K agents, each with (N_k, L, 2) trajectory array
        radius: Spatial threshold for stops
        min_duration: Minimum samples to qualify as stop
    
    Returns:
        List of K lists, each containing N_k arrays of shape (num_stops, 4)
        with columns: [start_time, end_time, lat, lon]
    """
    return [[detect_traj_stops(traj, radius, min_duration) for traj in agent]
            for agent in agents]


def detect_traj_stops(traj, radius, min_duration):
    """Detect stops in a single trajectory."""
    valid = np.isfinite(traj).all(axis=1)
    if valid.sum() < min_duration:
        return np.empty((0, 4))
    
    pts, idx, stops, i = traj[valid], np.where(valid)[0], [], 0
    
    while i < len(pts):
        dists = np.linalg.norm(pts[i:] - pts[i], axis=1)
        in_radius = dists <= radius
        run_len = np.argmin(in_radius) if not in_radius.all() else len(in_radius)
        
        if run_len >= min_duration:
            center = pts[i:i+run_len].mean(axis=0)
            stops.append([idx[i], idx[i+run_len-1], center[0], center[1]])
            i += run_len
        else:
            i += 1
    
    return np.array(stops) if stops else np.empty((0, 4))


def cluster_stops(stops, eps):
    """Cluster stops within eps distance."""
    if len(stops) == 0:
        return []
    
    coords, visited, clusters = stops[:, 2:4], np.zeros(len(stops), bool), []
    
    for i in range(len(stops)):
        if visited[i]:
            continue
        c = np.where(np.linalg.norm(coords - coords[i], axis=1) <= eps)[0]
        visited[c] = True
        clusters.append(c)
    
    return clusters


def unique_stops(train, test, eps):
    """Extract clustered centroids of test stops not in train."""
    if len(test) == 0:
        return np.empty((0, 2))
    
    new = test if len(train) == 0 else test[
        (np.linalg.norm(test[:, None, 2:4] - train[None, :, 2:4], axis=2) > eps).all(1)
    ]
    
    return (np.array([new[c, 2:4].mean(0) for c in cluster_stops(new, eps)]) 
            if len(new) else np.empty((0, 2)))
