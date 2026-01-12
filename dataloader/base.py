class Agent:
    def __init__(self, agent_id, anomalous=False):
        pass

    """
    Returns array of (t, lat, long) triples 
    """

    def samples(self, context="train"):
        pass


class Dataset:
    def __init__(self, **kwargs):
        pass

    def agent(self, agent_id):
        pass

    """
    Returns a length and an iterable of agents which satisfy all lambda 
    functions in filters
    """

    def agents(self, filters=[]):
        return 0, None
