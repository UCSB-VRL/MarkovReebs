import numpy as np

def extract_stop_points(trajectories, epsilon):
    """
    Extract stop points from trajectories.
    
    Args:
        trajectories: (N, L, 2) array of N trajectories
        epsilon: spatial threshold for stop detection
        
    Returns:
        List of N arrays, each (M, 4) with columns [x, y, start_idx, end_idx]
    """
    N, L = trajectories.shape[:2]
    result = []
    
    for traj in trajectories:
        dists = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
        is_stop = np.concatenate([[True], dists <= epsilon])
        
        changes = np.diff(np.concatenate([[0], is_stop.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            result.append(np.empty((0, 4)))
            continue
        
        stop_points = np.array([
            np.mean(traj[starts[i]:ends[i]], axis=0).tolist() + [starts[i], ends[i]-1]
            for i in range(len(starts))
        ])
        
        mask = np.isfinite(stop_points[:, 0]) & np.isfinite(stop_points[:, 1])
        result.append(stop_points[mask].T)
    
    return result
