from .base import Dataset, Agent

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, date

from paths import geolife_path


class GeolifeAgent(Agent):
    def __init__(self, agent_id):
        self.id = agent_id
        self.data = {}
        self.anomalous = False

    def samples(self, context="train"):
        return self.data[context]


class Geolife(Dataset):
    """
    split date -- determines how train/test split is constructed. if the split
    date is none, then all the samples are placed under the "train" split
    rate -- time between samples (in seconds) to use with interpolation
    dropout -- time between samples in seconds until we should consider the
    agent to have "disappeared"
    """

    def __init__(
        self,
        path=None,
        split_date=(2009, 5, 1),
        rate=10,
        dropout=30,
        **kwargs,
    ):
        if path is None:
            path = geolife_path()

        self.path = Path(path) / "Data"
        assert self.path.exists(), "Path does not exist."

        # store the list of anomalous agents
        self.agent_ids = {
            int(agent_id.name): agent_id
            for agent_id in self.path.iterdir()
            if self.path.is_dir()
        }

        self.agent_count = len(self.agent_ids)
        self.anomalous_agents = {}

        if split_date is not None:
            self.split_date = date(*split_date)
        else:
            # set an arbitrarily large split date
            self.split_date = date(9999, 12, 31)

        self.rate = rate
        self.dropout = dropout

    def __resample_df(self, df):
        """
        df: DataFrame with ['Latitude', 'Longitude', 'Time'] over multiple days.
        self.rate:    sampling interval in seconds
        self.dropout: max gap (in seconds) to interpolate across
        """

        df = df.copy()
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values("Time").set_index("Time")

        freq = f"{int(self.rate)}s"  # use 's' (lowercase) for seconds

        def _resample_one_day(day_df):
            # Remove duplicate timestamps (keep first or last occurrence)
            day_df = day_df[~day_df.index.duplicated(keep="first")]

            day = day_df.index[0].normalize()
            full_index = pd.date_range(
                start=day,
                end=day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                freq=freq,
            )

            out = day_df.reindex(full_index).interpolate(
                method="time", limit_direction="both"
            )[["Latitude", "Longitude"]]

            gaps = (
                day_df.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
            )
            times = day_df.index.to_numpy()

            intervals = [
                (times[i - 1], times[i])
                for i in range(1, len(times))
                if gaps[i] > self.dropout
            ]

            if intervals:
                idx_vals = out.index.to_numpy()
                long_gap_mask = np.logical_or.reduce(
                    [(idx_vals > t0) & (idx_vals < t1) for t0, t1 in intervals]
                )
                out.loc[long_gap_mask, :] = np.inf

            out = out.mask(out.isna(), np.inf)
            return out

        resampled = (
            pd.concat(
                [
                    _resample_one_day(day_df)
                    for _, day_df in df.groupby(df.index.normalize())
                ]
            )
            .reset_index()
            .rename(columns={"index": "Time"})
        )

        return resampled

    def agent(self, agent_id):
        agent = GeolifeAgent(agent_id)

        agent_path = self.path / self.agent_ids[agent_id] / "Trajectory"

        train_files = [
            file
            for file in Path(agent_path).glob("*.plt")
            if datetime.strptime(file.stem, "%Y%m%d%H%M%S").date() < self.split_date
        ]
        test_files = [
            file
            for file in Path(agent_path).glob("*.plt")
            if datetime.strptime(file.stem, "%Y%m%d%H%M%S").date() >= self.split_date
        ]

        df_schema = [
            "Latitude",
            "Longitude",
            "dummy",
            "altitude",
            "serial",
            "date",
            "time",
        ]

        if len(train_files) > 0:
            train_df = pd.concat(
                [
                    pd.read_csv(file, skiprows=6, header=None, names=df_schema)
                    for file in train_files
                ],
                ignore_index=True,
            )
        else:
            train_df = None

        if len(test_files) > 0:
            test_df = pd.concat(
                [
                    pd.read_csv(file, skiprows=6, header=None, names=df_schema)
                    for file in test_files
                ],
                ignore_index=True,
            )
        else:
            test_df = None

        for split_name, split_df in (("train", train_df), ("test", test_df)):
            if split_df is None:
                print("[WARNING] no data for", split_name)
                continue

            split_df["Time"] = pd.to_datetime(split_df["date"] + " " + split_df["time"])
            reduced_df = split_df[["Latitude", "Longitude", "Time"]]

            # Note -- we may need to experiment with the resampled df.
            resampled_df = self.__resample_df(reduced_df)

            agent.data[split_name] = np.array(
                [
                    g[["Latitude", "Longitude"]].to_numpy()
                    for _, g in resampled_df.groupby(
                        resampled_df["Time"].dt.normalize()
                    )
                ]
            )

        return agent

    """
    Returns a length and an iterable of agents which satisfy all lambda 
    functions in filters
    """

    def agents(self, filters=[]):
        return 0, None


if __name__ == "__main__":
    sim = Geolife()
    agent = sim.agent(0)
