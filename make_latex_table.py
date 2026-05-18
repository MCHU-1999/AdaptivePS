import os
import json
import numpy as np

def load_results(base_dir):
    results = {}
    if not os.path.exists(base_dir):
        return results
    
    for scan in os.listdir(base_dir):
        scan_path = os.path.join(base_dir, scan)
        if not os.path.isdir(scan_path):
            continue
        
        result_file = os.path.join(scan_path, 'results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[scan] = data
    return results

def main():
    base_dir = '/Users/mchu/Documents/TUD/Thesis/PlanarSplatting'
    vanilla_dir = os.path.join(base_dir, 'Vanilla/eval_results')
    adaptive_dir = os.path.join(base_dir, 'AdaptivePS/eval_results')
    
    vanilla_results = load_results(vanilla_dir)
    adaptive_results = load_results(adaptive_dir)
    
    # Load runtimes
    runtime_dir = os.path.join(base_dir, 'evaluation/runtime_logs')
    vanilla_rt_file = os.path.join(runtime_dir, 'vanilla.json')
    adaptive_rt_file = os.path.join(runtime_dir, 'adaptiveps.json')
    da3_rt_file = os.path.join(runtime_dir, 'da3.json')
    sam_rt_file = os.path.join(runtime_dir, 'sam.json')
    
    vanilla_rt = {}
    if os.path.exists(vanilla_rt_file):
        with open(vanilla_rt_file, 'r') as f:
            vanilla_rt = json.load(f)
            
    adaptive_rt = {}
    if os.path.exists(adaptive_rt_file):
        with open(adaptive_rt_file, 'r') as f:
            adaptive_rt = json.load(f)
            
    da3_rt = {}
    if os.path.exists(da3_rt_file):
        with open(da3_rt_file, 'r') as f:
            da3_rt = json.load(f)
            
    sam_rt = {}
    if os.path.exists(sam_rt_file):
        with open(sam_rt_file, 'r') as f:
            sam_rt = json.load(f)
    
    # Get common scans, sort them numerically
    scans = set(vanilla_results.keys()).intersection(set(adaptive_results.keys()))
    scans = sorted(list(scans), key=lambda x: int(x.replace('scan', '')))
    
    metrics = ['overall', 'fscore@2.0mm', 'runtime']
    
    # Initialize arrays for means
    vanilla_means = {m: [] for m in metrics}
    adaptive_means = {m: [] for m in metrics}
    
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\small")
    print("% Define a shortcut for the best cells (background + bold text)")
    print("\\newcommand{\\best}[1]{\\cellcolor{cyan!20}\\textbf{#1}}")
    print("")
    print("\\begin{tabularx}{\\linewidth}{l *{6}{>{\\centering\\arraybackslash}X}}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Chamfer Dist. $\\downarrow$} & \\multicolumn{2}{c}{F1-score @ 2mm $\\uparrow$} & \\multicolumn{2}{c}{Runtime (s) $\\downarrow$} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print("Scan & Baseline & AdaptivePS & Baseline & AdaptivePS & Baseline & AdaptivePS \\\\")
    print("\\midrule")
    
    def format_pair(val_b, val_a, is_lower_better=True):
        if val_b is None and val_a is None:
            return "-", "-"
        str_b = f"{val_b:.2f}" if val_b is not None else "-"
        str_a = f"{val_a:.2f}" if val_a is not None else "-"
        
        if val_b is not None and val_a is not None:
            if is_lower_better:
                if val_b < val_a:
                    str_b = f"\\best{{{str_b}}}"
                elif val_a < val_b:
                    str_a = f"\\best{{{str_a}}}"
            else:
                if val_b > val_a:
                    str_b = f"\\best{{{str_b}}}"
                elif val_a > val_b:
                    str_a = f"\\best{{{str_a}}}"
        return str_b, str_a
    
    for scan in scans:
        v_res = vanilla_results.get(scan, {})
        a_res = adaptive_results.get(scan, {})
        
        v_rt_data = vanilla_rt.get(scan, {})
        v_rt_val = v_rt_data.get('total_s', None) if isinstance(v_rt_data, dict) else v_rt_data
        
        a_rt_data = adaptive_rt.get(scan, None)
        a_rt_val = a_rt_data.get('total_s', None) if isinstance(a_rt_data, dict) else a_rt_data
        
        da3_rt_val = da3_rt.get(scan, 0.0)
        sam_rt_val = sam_rt.get(scan, 0.0)
        
        baseline_rt = v_rt_val + da3_rt_val if v_rt_val is not None else None
        adaptive_rt_total = a_rt_val + da3_rt_val + sam_rt_val if a_rt_val is not None else None
        
        val_cd_v = v_res.get('overall', None)
        val_cd_a = a_res.get('overall', None)
        
        val_fs_v = v_res.get('fscore@2.0mm', None)
        val_fs_a = a_res.get('fscore@2.0mm', None)
        
        if val_cd_v is not None: vanilla_means['overall'].append(val_cd_v)
        if val_cd_a is not None: adaptive_means['overall'].append(val_cd_a)
        
        if val_fs_v is not None: vanilla_means['fscore@2.0mm'].append(val_fs_v)
        if val_fs_a is not None: adaptive_means['fscore@2.0mm'].append(val_fs_a)
        
        if baseline_rt is not None: vanilla_means['runtime'].append(baseline_rt)
        if adaptive_rt_total is not None: adaptive_means['runtime'].append(adaptive_rt_total)
        
        str_cd_b, str_cd_a = format_pair(val_cd_v, val_cd_a, is_lower_better=True)
        str_fs_b, str_fs_a = format_pair(val_fs_v, val_fs_a, is_lower_better=False)
        str_rt_b, str_rt_a = format_pair(baseline_rt, adaptive_rt_total, is_lower_better=True)
        
        row = [scan.replace('scan', ''), str_cd_b, str_cd_a, str_fs_b, str_fs_a, str_rt_b, str_rt_a]
        print(" & ".join(row) + " \\\\")
        
    print("\\midrule")
    
    # Compute and print means
    mean_cd_v = np.mean(vanilla_means['overall']) if vanilla_means['overall'] else None
    mean_cd_a = np.mean(adaptive_means['overall']) if adaptive_means['overall'] else None
    mean_fs_v = np.mean(vanilla_means['fscore@2.0mm']) if vanilla_means['fscore@2.0mm'] else None
    mean_fs_a = np.mean(adaptive_means['fscore@2.0mm']) if adaptive_means['fscore@2.0mm'] else None
    mean_rt_v = np.mean(vanilla_means['runtime']) if vanilla_means['runtime'] else None
    mean_rt_a = np.mean(adaptive_means['runtime']) if adaptive_means['runtime'] else None
    
    m_cd_b, m_cd_a = format_pair(mean_cd_v, mean_cd_a, is_lower_better=True)
    m_fs_b, m_fs_a = format_pair(mean_fs_v, mean_fs_a, is_lower_better=False)
    m_rt_b, m_rt_a = format_pair(mean_rt_v, mean_rt_a, is_lower_better=True)
    
    mean_row = ["Mean", m_cd_b, m_cd_a, m_fs_b, m_fs_a, m_rt_b, m_rt_a]
    print(" & ".join(mean_row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\caption{Quantitative comparison on the DTU dataset. Chamfer Distance is measured in mm. Shaded cells indicate the top performance for each metric.}")
    print("\\label{tab:dtu_results}")
    print("\\end{table}")

if __name__ == '__main__':
    main()
