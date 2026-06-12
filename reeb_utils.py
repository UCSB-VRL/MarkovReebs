import numpy as np


def finite_euclidean_distance(x, y):
    return 0 if np.isinf(x).any() and np.isinf(y).any() else np.linalg.norm(x - y)


def identity_orderer(x):
    return x


def equality_equivalence(x, y):
    return x == y


def make_pickle_safe(reeb):
    reeb.dist = finite_euclidean_distance
    reeb.orderer = identity_orderer
    reeb.equivalence = equality_equivalence
    return reeb
