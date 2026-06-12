# MarkovReebs

MarkovReebs contains experiments for generating synthetic mobility trajectories from trajectory datasets using our Reeb graph-based generation methods and baseline methods.

The main workflow is:

1. Load Urban Anomalies or Geolife trajectories.
2. Build a multi-agent Reeb graph (MARG) from training trajectories.
3. Generate per-agent trajectories with SRG, HRG, or Markov-chain baselines.
4. Evaluate generated trajectories against held-out trajectories.

## Setup

Install dependencies with `uv`:

```sh
uv sync
```

Download both of the necessary datasets. Set dataset locations in the environment or in `.envrc`:

```sh
export URBAN_ANOMALIES_PATH=/path/to/UrbanAnomalies
export GEOLIFE_PATH="/path/to/Geolife 1.3"
```

`DATAPATH` is also accepted as a fallback for Urban Anomalies. If these variables are not set, the loaders fall back to the historical defaults under `/data/Datasets/`. 

## Generate Outputs

Build a MARG pickle:

```sh
uv run python marg.py
```

Generate single-agent Reeb graph outputs:

```sh
uv run python srg.py
```

Generate hybrid Reeb graph outputs:

```sh
uv run python hrg.py
```

## Evaluate

Run the evaluation script after generating the required files in `outputs/`:

```sh
uv run python eval.py
```

`eval.py` loads method outputs, optionally trims Geolife trajectories, computes agent and population metrics, and prints the result dictionary.

It also writes the captured results to:

```text
outputs/table.log
```

Then it immediately runs the table formatter and writes:

```text
outputs/generated_results.typ
```

## Repository Layout

`dataloader/` contains dataset adapters for Urban Anomalies and Geolife.

`marg.py`, `srg.py`, and `hrg.py` build multi-agent, single-agent, and hybrid Reeb graph outputs.

`baseline.py` and `baseline/` implement the Markov-chain baseline.

`evaluation/` and `eval.py` compute metrics and table-ready summaries.

`visualize.py` and `visualizations/` contain exploratory plotting utilities.
