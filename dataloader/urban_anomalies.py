from .base import Dataset, Agent

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from paths import urban_anomalies_path


class UAAgent(Agent):
    def __init__(self, agent_id, anomalous=False):
        self.id = agent_id
        self.anomalous = anomalous
        self.data = {}

    def samples(self, context="train"):
        return self.data[context]


class UrbanAnomalies(Dataset):
    """
    method -- centralized, infectious or location
    type -- combined, hunger, interest, social or work
    location -- atlanta or berlin
    """

    def __init__(
        self,
        path=None,
        method="centralized",
        anomaly_type="combined",
        location="atlanta",
        manual_path=None,
        **kwargs,
    ):
        # TODO: compute non-centralized datasets
        if path is None:
            path = urban_anomalies_path()

        if manual_path is None:
            self.path = Path(path) / method / f"{location}_{anomaly_type}_outliers"
        else:
            self.path = Path(manual_path)

        assert self.path.exists(), "Path does not exist."

        # load labels.json to get a list of anomalies
        # TODO: categorize anomalies by intensity
        with open(self.path / "labels.json", "r") as f:
            self.labels = json.load(f)

        with open(self.path / "info.json", "r") as f:
            self.info = json.load(f)

        # store the list of anomalous agents
        self.agent_count = self.info["number_of_agents"]
        self.anomalous_agents = {int(agent_id) for agent_id in self.labels}

        self.data = {}

    def agent(self, agent_id):
        if "train" not in self.data:
            train_path = self.path / "trajectories_train.tsv"

            # auto-extract tsv files
            if not train_path.exists():
                train_zip = self.path / "trajectories_train.zip"
                assert train_zip.exists(), "Train zip does not exist"
                with zipfile.ZipFile(train_zip, "r") as zf:
                    zf.extractall(self.path)

            self.data["train"] = pd.read_csv(train_path)

        if "test" not in self.data:
            test_path = self.path / "trajectories_test.tsv"
            if not test_path.exists():
                test_zip = self.path / "trajectories_test.zip"
                assert test_zip.exists(), "Test zip does not exist"
                with zipfile.ZipFile(test_zip, "r") as zf:
                    zf.extractall(self.path)

            self.data["test"] = pd.read_csv(test_path)

        train_data = self.data["train"]
        test_data = self.data["test"]

        agent = UAAgent(
            agent_id, anomalous=True if agent_id in self.anomalous_agents else False
        )

        # get agent data
        agent_train = train_data[train_data["AgentID"] == agent_id]
        agent_test = test_data[test_data["AgentID"] == agent_id]

        # chunk into by day matrices
        agent_train_times = pd.to_datetime(agent_train["Time"])
        agent_train_by_day = [
            g for _, g in agent_train.groupby(agent_train_times.dt.date)
        ]

        agent_test_times = pd.to_datetime(agent_test["Time"])
        agent_test_by_day = [g for _, g in agent_test.groupby(agent_test_times.dt.date)]

        N = len(agent_train_by_day)
        L = agent_train_by_day[0].shape[0]
        D = 2

        train_tensor = np.empty((N, L, D))
        for day, df in enumerate(agent_train_by_day):
            train_tensor[day] = df[["Latitude", "Longitude"]].to_numpy()

        M = len(agent_test_by_day)
        assert agent_test_by_day[0].shape[0] == L

        test_tensor = np.empty((N, L, D))
        for day, df in enumerate(agent_test_by_day):
            test_tensor[day] = df[["Latitude", "Longitude"]].to_numpy()

        agent.data["train"] = train_tensor
        agent.data["test"] = test_tensor

        return agent

    """
    Returns a length and an iterable of agents which satisfy all lambda 
    functions in filters
    """

    def agents(self, filters=[]):
        return 0, None
