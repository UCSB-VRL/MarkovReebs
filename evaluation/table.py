"""
Input: a dictionary containing all of the data
Output: prints a typst table into the terminal
"""
import ast
import re

def generate_typst_from_dict(data_list, dataset_name="ua"):
    # 1. Map metrics to their M2, HRG, and SRG values
    # Standard metrics found in the log: 'Distance Traveled', 'Duration of Movement', etc.
    metrics_map = {}
    
    for r in data_list:
        # Identify metrics by excluding metadata keys 
        metric_keys = [k for k in r.keys() if k not in 
                       ['Evaluation Method', 'Generation Method', 'Source']]
        
        for m in metric_keys:
            if m not in metrics_map:
                metrics_map[m] = {'M2': {}, 'HRG': {}, 'SRG': {}}
            
            # Identify the generation method; use 'Source' to identify the M2 baseline 
            gen_key = 'M2' if r.get('Source') == 'M2' else r['Generation Method'].upper()
            eval_type = r['Evaluation Method'] # 'agent' or 'population' 
            
            metrics_map[m][gen_key][eval_type] = float(r[m])

    # 2. Build the Typst string
    caption = "Urban Anomalies" if dataset_name == "ua" else "Geolife"
    typst = f'#figure(caption: [{caption}])[\n'
    typst += '  #table(\n    columns: 4,\n    align: center + bottom,\n    table.hline(),\n'
    typst += '    [Evaluation Statistic], [Generation Method], [Agent Score], [Population Score],\n    table.hline(),\n'

    # 3. Sort metrics alphabetically as requested
    for metric in sorted(metrics_map.keys()):
        m_data = metrics_map[metric]
        m2_agent = m_data['M2']['agent']
        m2_pop = m_data['M2']['population']
        
        # Internal helper for bolding logic based on proximity to M2
        def format_val(method, eval_type):
            val = m_data[method][eval_type]
            other = 'SRG' if method == 'HRG' else 'HRG'
            other_val = m_data[other][eval_type]
            base = m2_agent if eval_type == 'agent' else m2_pop
            
            # Bold if closer to M2 baseline, or if it's an exact tie [cite: 1, 3]
            is_closer = abs(val - base) <= abs(other_val - base)
            formatted = f"{val:.5f}"
            return f"[* {formatted} *]" if is_closer else f"[{formatted}]"

        # Table rows construction
        typst += f'    table.cell(rowspan: 3, align: center + horizon)[{metric}], \n'
        # Row 1: M2 Benchmark 
        typst += f'      [M2], [{m2_agent:.5f}], [{m2_pop:.5f}],\n'
        # Row 2: HRG [cite: 3]
        typst += f'      [HRG], {format_val("HRG", "agent")}, {format_val("HRG", "population")},\n'
        # Row 3: SRG [cite: 2]
        typst += f'      [SRG], {format_val("SRG", "agent")}, {format_val("SRG", "population")},\n'
        typst += '    table.hline(),\n'
        
    typst += '  )\n]'
    return typst

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

    # 2. Process each dataset
    output = []
    for ds_name, records in datasets.items():
        # Map metrics to their M2, HRG, and SRG values
        metrics_map = {}
        for r in records:
            metric_names = [k for k in r.keys() if k not in ['Evaluation Method', 'Generation Method', 'Source']]
            for m in metric_names:
                if m not in metrics_map: metrics_map[m] = {}
                gen_method = r['Generation Method']
                eval_type = r['Evaluation Method'] # 'agent' or 'population'
                
                # If Source is M2, treat as benchmark
                key = 'M2' if r['Source'] == 'M2' else gen_method.upper()
                if key not in metrics_map[m]: metrics_map[m][key] = {}
                metrics_map[m][key][eval_type] = float(r[m])

        # 3. Start building the Typst string
        caption = "Urban Anomalies" if ds_name == "ua" else "Geolife"
        typst = f'#figure(caption: [{caption}])[\n'
        typst += '  #table(\n    columns: 4,\n    align: center + bottom,\n    table.hline(),\n'
        typst += '    [Evaluation Statistic], [Generation Method], [Agent Score], [Population Score],\n    table.hline(),\n'

        # Sort metrics alphabetically
        for metric in sorted(metrics_map.keys()):
            m_data = metrics_map[metric]
            m2_a, m2_p = m_data['M2']['agent'], m_data['M2']['population']
            
            # Helper for bolding logic
            def get_val(method, eval_type):
                val = m_data[method][eval_type]
                other_method = 'SRG' if method == 'HRG' else 'HRG'
                other_val = m_data[other_method][eval_type]
                base = m2_a if eval_type == 'agent' else m2_p
                
                diff_this = abs(val - base)
                diff_other = abs(other_val - base)
                
                formatted = f"{val:.5f}"
                return f"[* {formatted} *]" if diff_this <= diff_other else f"[{formatted}]"

            # Table rows for the metric
            typst += f'    table.cell(rowspan: 3, align: center + horizon)[{metric}], \n'
            # Row 1: M2
            typst += f'      [M2], [{m2_a:.5f}], [{m2_p:.5f}],\n'
            # Row 2: HRG
            typst += f'      [HRG], {get_val("HRG", "agent")}, {get_val("HRG", "population")},\n'
            # Row 3: SRG
            typst += f'      [SRG], {get_val("SRG", "agent")}, {get_val("SRG", "population")},\n'
            typst += '    table.hline(),\n'
            
        typst += '  )\n]'
        output.append(typst)
    
    return "\n\n".join(output)

if __name__ == "__main__":
    log_path = "../outputs/table.log"
    print(generate_typst_table(log_path))

