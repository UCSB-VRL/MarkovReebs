"""
Input: a dictionary containing all of the data
Output: prints a typst table into the terminal
"""
import ast
from pathlib import Path
import re


METADATA_KEYS = {"Evaluation Method", "Generation Method", "Source", "Epsilon"}


def _method_key(row):
    if row.get("Source") == "M2":
        return "M2"

    method = row["Generation Method"].upper()
    if "Epsilon" in row:
        return f"{method} (epsilon={row['Epsilon']})"

    return method


def _records_to_typst(records, dataset_name="ua"):
    metrics_map = {}

    for row in records:
        metric_names = [key for key in row if key not in METADATA_KEYS]
        method = _method_key(row)
        eval_type = row["Evaluation Method"]

        for metric in metric_names:
            metrics_map.setdefault(metric, {}).setdefault(method, {})[eval_type] = float(row[metric])

    caption = "Urban Anomalies" if dataset_name == "ua" else "Geolife"
    typst = f'#figure(caption: [{caption}])[\n'
    typst += '  #table(\n    columns: 4,\n    align: center + bottom,\n    table.hline(),\n'
    typst += '    [Evaluation Statistic], [Generation Method], [Agent Score], [Population Score],\n    table.hline(),\n'

    for metric in sorted(metrics_map):
        methods = metrics_map[metric]
        ordered_methods = ["M2"] + sorted(method for method in methods if method != "M2")
        base = methods.get("M2", {})
        candidates = [method for method in ordered_methods if method != "M2"]

        best = {}
        for eval_type in ("agent", "population"):
            if eval_type in base and candidates:
                best[eval_type] = min(
                    candidates,
                    key=lambda method: abs(methods[method].get(eval_type, float("inf")) - base[eval_type]),
                )

        typst += f'    table.cell(rowspan: {len(ordered_methods)}, align: center + horizon)[{metric}], \n'
        for method in ordered_methods:
            values = methods[method]
            cells = []
            for eval_type in ("agent", "population"):
                value = values.get(eval_type)
                if value is None:
                    cells.append("[-]")
                    continue

                formatted = f"{value:.5f}"
                if method != "M2" and best.get(eval_type) == method:
                    cells.append(f"[* {formatted} *]")
                else:
                    cells.append(f"[{formatted}]")

            typst += f'      [{method}], {cells[0]}, {cells[1]},\n'

        typst += '    table.hline(),\n'

    typst += '  )\n]'
    return typst


def generate_typst_from_dict(data_list, dataset_name="ua"):
    return _records_to_typst(data_list, dataset_name)

def generate_typst_table(log_path):
    with open(log_path, "r") as file:
        log_content = file.read()

    # 1. Parse datasets from the log
    datasets = {}
    current_dataset = None
    
    # Split by dataset headers
    sections = re.split(r'Dataset: ', log_content)
    for section in sections[1:]:
        lines = section.split('\n')
        dataset_name = lines[0].strip()
        # Find the list of dictionaries
        list_str = section[len(dataset_name):].strip()
        # Clean up np.float64 wrappers
        list_str = re.sub(r'np\.float64\((.*?)\)', r'\1', list_str)
        # Handle source markers like 
        # list_str = re.sub(r'\'', '', list_str)
        
        try:
            data_list = ast.literal_eval(list_str)
            datasets[dataset_name] = data_list
        except:
            continue

    return "\n\n".join(
        _records_to_typst(records, dataset_name=dataset_name)
        for dataset_name, records in datasets.items()
    )

if __name__ == "__main__":
    log_path = Path("outputs/table.log")
    if not log_path.exists():
        log_path = Path(__file__).resolve().parents[1] / "outputs" / "table.log"
    print(generate_typst_table(log_path))
