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
    import re
    base_dir = '/Users/mchu/Documents/TUD/Thesis/PlanarSplatting'
    baseline_dir  = os.path.join(base_dir, 'Baseline/eval_results')
    adaptive_dir = os.path.join(base_dir, 'AdaptivePS/eval_results')

    baseline_results  = load_results(baseline_dir)
    adaptive_results = load_results(adaptive_dir)

    # Load runtimes
    runtime_dir      = os.path.join(base_dir, 'evaluation/runtime_logs')
    def _load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}
    baseline_rt  = _load_json(os.path.join(runtime_dir, 'baseline.json'))
    adaptive_rt = _load_json(os.path.join(runtime_dir, 'adaptiveps.json'))
    da3_rt      = _load_json(os.path.join(runtime_dir, 'da3.json'))
    sam_rt      = _load_json(os.path.join(runtime_dir, 'sam.json'))

    # Plane-count helpers (reused from main_ablation logic)
    def find_latest_run(scene_dir):
        try:
            entries = sorted([e for e in os.listdir(scene_dir)
                              if os.path.isdir(os.path.join(scene_dir, e))])
            return os.path.join(scene_dir, entries[-1]) if entries else None
        except Exception:
            return None

    def get_planes_for_scan(runs_base, scan_name):
        scan_path = os.path.join(base_dir, runs_base, scan_name)
        # scan folders may have a _APS suffix — pick the first matching dir
        if not os.path.isdir(scan_path):
            candidates = [d for d in os.listdir(os.path.join(base_dir, runs_base))
                          if d.startswith(scan_name)]
            if not candidates:
                return None
            scan_path = os.path.join(base_dir, runs_base, sorted(candidates)[0])
        latest = find_latest_run(scan_path)
        if latest is None:
            return None
        log = os.path.join(latest, 'train.log')
        pattern = re.compile(r"number of planar instances = (\d+)")
        final = None
        try:
            with open(log) as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        final = int(m.group(1))
        except Exception:
            pass
        return final

    # Common scans, sorted numerically
    scans = set(baseline_results.keys()).intersection(set(adaptive_results.keys()))
    scans = sorted(list(scans), key=lambda x: int(x.replace('scan', '')))

    metrics = ['planes', 'overall', 'fscore@2.0mm', 'runtime']
    baseline_means  = {m: [] for m in metrics}
    adaptive_means = {m: [] for m in metrics}

    def format_pair(val_b, val_a, is_lower_better=True, fmt=".2f", suffix=""):
        if val_b is None and val_a is None:
            return "-", "-"
        str_b = f"{val_b:{fmt}}{suffix}" if val_b is not None else "-"
        str_a = f"{val_a:{fmt}}{suffix}" if val_a is not None else "-"
        if val_b is not None and val_a is not None:
            if is_lower_better:
                best_b = val_b < val_a
            else:
                best_b = val_b > val_a
            if val_b != val_a:
                if best_b:
                    str_b = f"\\best{{{str_b}}}"
                else:
                    str_a = f"\\best{{{str_a}}}"
        return str_b, str_a

    # Header
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\footnotesize")
    print("\\newcommand{\\best}[1]{\\cellcolor{red!25}\\textbf{#1}}")
    print("")
    print("\\caption{Quantitative comparison on the DTU dataset. Chamfer Distance is measured in millimeters. Shaded cells indicate the top performance for each metric.}")
    print("\\label{tab:dtu_results}")
    print("\\begin{tabularx}{\\linewidth}{l *{8}{>{\\centering\\arraybackslash}X}}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Final Planes $\\downarrow$} & \\multicolumn{2}{c}{CD (mm)$\\downarrow$} & \\multicolumn{2}{c}{F1-Score @ 2mm $\\uparrow$} & \\multicolumn{2}{c}{Runtime $\\downarrow$} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}")
    print("Scan & Baseline & AdaptivePS & Baseline & AdaptivePS & Baseline & AdaptivePS & Baseline & AdaptivePS \\\\")
    print("\\midrule")

    for scan in scans:
        v_res = baseline_results.get(scan, {})
        a_res = adaptive_results.get(scan, {})

        # Planes
        v_planes = get_planes_for_scan('Baseline/DTU-Building', scan)
        a_planes = get_planes_for_scan('AdaptivePS/DTU-Building', scan)

        # Runtime
        v_rt_data = baseline_rt.get(scan, {})
        v_rt_val  = v_rt_data.get('total_s', None) if isinstance(v_rt_data, dict) else v_rt_data
        a_rt_data = adaptive_rt.get(scan, None)
        a_rt_val  = a_rt_data.get('total_s', None) if isinstance(a_rt_data, dict) else a_rt_data
        da3_val   = da3_rt.get(scan, 0.0)
        sam_val   = sam_rt.get(scan, 0.0)
        # baseline_rt  = v_rt_val + da3_val if v_rt_val is not None else None
        baseline_rt  = v_rt_val if v_rt_val is not None else None
        adaptive_rt_total = a_rt_val + da3_val + sam_val if a_rt_val is not None else None

        val_cd_v = v_res.get('overall', None)
        val_cd_a = a_res.get('overall', None)
        val_fs_v = v_res.get('fscore@2.0mm', None)
        val_fs_a = a_res.get('fscore@2.0mm', None)

        # Accumulate for means
        for lst, val in [(baseline_means['planes'], v_planes),
                         (adaptive_means['planes'], a_planes),
                         (baseline_means['overall'], val_cd_v),
                         (adaptive_means['overall'], val_cd_a),
                         (baseline_means['fscore@2.0mm'], val_fs_v),
                         (adaptive_means['fscore@2.0mm'], val_fs_a),
                         (baseline_means['runtime'], baseline_rt),
                         (adaptive_means['runtime'], adaptive_rt_total)]:
            if val is not None:
                lst.append(val)

        str_pl_b = f"{v_planes:.0f}" if v_planes is not None else "-"
        str_pl_a = f"{a_planes:.0f}" if a_planes is not None else "-"
        str_cd_b, str_cd_a = format_pair(val_cd_v, val_cd_a, is_lower_better=True)
        str_fs_b, str_fs_a = format_pair(val_fs_v, val_fs_a, is_lower_better=False)
        str_rt_b, str_rt_a = format_pair(baseline_rt, adaptive_rt_total, is_lower_better=True, fmt=".0f", suffix="s")

        row = [scan.replace('scan', ''),
               str_pl_b, str_pl_a,
               str_cd_b, str_cd_a,
               str_fs_b, str_fs_a,
               str_rt_b, str_rt_a]
        print(" & ".join(row) + " \\\\")

    print("\\midrule")

    def _mean(lst): return np.mean(lst) if lst else None

    v_mean_pl = _mean(baseline_means['planes'])
    a_mean_pl = _mean(adaptive_means['planes'])
    m_pl_b = f"{v_mean_pl:.1f}" if v_mean_pl is not None else "-"
    m_pl_a = f"{a_mean_pl:.1f}" if a_mean_pl is not None else "-"
    m_cd_b, m_cd_a = format_pair(_mean(baseline_means['overall']),      _mean(adaptive_means['overall']),      is_lower_better=True)
    m_fs_b, m_fs_a = format_pair(_mean(baseline_means['fscore@2.0mm']), _mean(adaptive_means['fscore@2.0mm']), is_lower_better=False)
    m_rt_b, m_rt_a = format_pair(_mean(baseline_means['runtime']),      _mean(adaptive_means['runtime']),      is_lower_better=True,  fmt=".0f", suffix="s")

    mean_row = ["Mean", m_pl_b, m_pl_a, m_cd_b, m_cd_a, m_fs_b, m_fs_a, m_rt_b, m_rt_a]
    print(" & ".join(mean_row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table}")

def main_chamfer_detailed():
    """Table comparing all 3 Chamfer components: Acc. (mean_d2s), Comp. (mean_s2d), Overall."""
    base_dir = '/Users/mchu/Documents/TUD/Thesis/PlanarSplatting'
    baseline_dir = os.path.join(base_dir, 'Baseline/eval_results')
    adaptive_dir = os.path.join(base_dir, 'AdaptivePS/eval_results')

    baseline_results = load_results(baseline_dir)
    adaptive_results = load_results(adaptive_dir)

    scans = set(baseline_results.keys()).intersection(set(adaptive_results.keys()))
    scans = sorted(list(scans), key=lambda x: int(x.replace('scan', '')))

    cd_metrics = ['mean_d2s', 'mean_s2d', 'overall']

    baseline_means = {m: [] for m in cd_metrics}
    adaptive_means = {m: [] for m in cd_metrics}

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

    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\small")
    print("% Define a shortcut for the best cells (background + bold text)")
    print("\\newcommand{\\best}[1]{\\cellcolor{cyan!20}\\textbf{#1}}")
    print("")
    print("\\begin{tabularx}{\\linewidth}{l *{6}{>{\\centering\\arraybackslash}X}}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Acc. $\\downarrow$} & \\multicolumn{2}{c}{Comp. $\\downarrow$} & \\multicolumn{2}{c}{Overall $\\downarrow$} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print("Scan & Baseline & AdaptivePS & Baseline & AdaptivePS & Baseline & AdaptivePS \\\\")
    print("\\midrule")

    for scan in scans:
        v_res = baseline_results.get(scan, {})
        a_res = adaptive_results.get(scan, {})

        vals = {}
        for m in cd_metrics:
            vals[f'v_{m}'] = v_res.get(m, None)
            vals[f'a_{m}'] = a_res.get(m, None)
            if vals[f'v_{m}'] is not None: baseline_means[m].append(vals[f'v_{m}'])
            if vals[f'a_{m}'] is not None: adaptive_means[m].append(vals[f'a_{m}'])

        s_d2s_b, s_d2s_a = format_pair(vals['v_mean_d2s'], vals['a_mean_d2s'])
        s_s2d_b, s_s2d_a = format_pair(vals['v_mean_s2d'], vals['a_mean_s2d'])
        s_ov_b,  s_ov_a  = format_pair(vals['v_overall'],  vals['a_overall'])

        row = [scan.replace('scan', ''), s_d2s_b, s_d2s_a, s_s2d_b, s_s2d_a, s_ov_b, s_ov_a]
        print(" & ".join(row) + " \\\\")

    print("\\midrule")

    def mean_or_none(lst):
        return np.mean(lst) if lst else None

    m_d2s_b, m_d2s_a = format_pair(mean_or_none(baseline_means['mean_d2s']), mean_or_none(adaptive_means['mean_d2s']))
    m_s2d_b, m_s2d_a = format_pair(mean_or_none(baseline_means['mean_s2d']), mean_or_none(adaptive_means['mean_s2d']))
    m_ov_b,  m_ov_a  = format_pair(mean_or_none(baseline_means['overall']),  mean_or_none(adaptive_means['overall']))

    mean_row = ["Mean", m_d2s_b, m_d2s_a, m_s2d_b, m_s2d_a, m_ov_b, m_ov_a]
    print(" & ".join(mean_row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\caption{Detailed Chamfer Distance breakdown on the DTU dataset. Acc.\\ (mean\\_d2s) and Comp.\\ (mean\\_s2d) are measured in mm. Shaded cells indicate top performance.}")
    print("\\label{tab:dtu_chamfer_detailed}")
    print("\\end{table}")

def main_ablation():
    """
    Generates the LaTeX ablation table comparing all ablation variants vs. the full model.
    Columns: Planes no., CD (overall), F1-Score.
    Rows: Normalswap, No1mesh, Nosplit, Notrim, Full model (AdaptivePS/DTU-Building).
    """
    import re

    BASE_DIR = '/Users/mchu/Documents/TUD/Thesis/PlanarSplatting'

    VARIANTS = [
        ("Replace normal source (to Metric3Dv2 \\citep{hu_metric3dv2_2024})", "Ablation/Normalswap/eval_results", "Ablation/Normalswap", "normal"),
        ("w/o Mesh post-processing", "Ablation/No1mesh/eval_results", "Ablation/No1mesh", "loo_first"),
        ("w/o Mask-Guided Densification \\& Pruning", "Ablation/Nosplit/eval_results", "Ablation/Nosplit", "loo"),
        ("w/o Final Mask-Guided Trim", "Ablation/Notrim/eval_results", "Ablation/Notrim", "loo"),
        ("Only Mesh post-processing", "Ablation/Only1mesh/eval_results", "Ablation/Only1mesh", "iso_first"),
        ("Only Mask-Guided Densification \\& Pruning", "Ablation/Onlysplit/eval_results", "Ablation/Onlysplit", "iso"),
        ("Only Final Mask-Guided Trim", "Ablation/Onlytrim/eval_results", "Ablation/Onlytrim", "iso"),
        ("All 3 modules disabled", "Ablation/Allnone/eval_results", "Ablation/Allnone", "all_out"),
        ("Full model", "AdaptivePS/eval_results", "AdaptivePS/DTU-Building", "full"),
    ]

    def find_latest_run(scene_dir):
        try:
            entries = sorted([e for e in os.listdir(scene_dir)
                              if os.path.isdir(os.path.join(scene_dir, e))])
            return os.path.join(scene_dir, entries[-1]) if entries else None
        except Exception:
            return None

    def get_final_planar_instances(log_file):
        pattern = re.compile(r"number of planar instances = (\d+)")
        final = None
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        final = int(m.group(1))
        except Exception:
            pass
        return final

    def mean_planes(runs_dir_abs):
        """Return mean plane count across all scans in a runs directory."""
        counts = []
        if not os.path.isdir(runs_dir_abs):
            return None
        for scan in sorted(os.listdir(runs_dir_abs)):
            scan_path = os.path.join(runs_dir_abs, scan)
            if not os.path.isdir(scan_path):
                continue
            latest = find_latest_run(scan_path)
            if latest is None:
                continue
            log = os.path.join(latest, 'train.log')
            n = get_final_planar_instances(log)
            if n is not None:
                counts.append(n)
        return np.mean(counts) if counts else None

    def mean_metric(eval_dir_abs, key):
        results = load_results(eval_dir_abs)
        vals = [v[key] for v in results.values() if key in v]
        return np.mean(vals) if vals else None

    # --- print table ---
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\footnotesize")
    print("\\caption{Ablation study on the DTU \\textit{building} subset (taking mean values). \"Red\", \"Orange\" and \"Yellow\" denote the top 1-3 results.}")
    print("\\label{tab:ablation_dtu}")
    print("\\begin{tabular}{l ccc}")
    print("\\toprule")
    print("Model Setting & Planes no. $\\downarrow$ & CD (mm)$\\downarrow$ & F1-score @ 2mm $\\uparrow$ \\\\")
    print("\\midrule")

    rows_data = []
    planes_vals = []
    cd_vals = []
    f1_vals = []

    for i, (label, eval_rel, runs_rel, group) in enumerate(VARIANTS):
        eval_abs = os.path.join(BASE_DIR, eval_rel)
        runs_abs = os.path.join(BASE_DIR, runs_rel)

        planes = mean_planes(runs_abs)
        cd     = mean_metric(eval_abs, 'overall')
        f1     = mean_metric(eval_abs, 'fscore@2.0mm')

        rows_data.append({
            'label': label,
            'planes': planes,
            'cd': cd,
            'f1': f1,
            'group': group
        })

        if planes is not None: planes_vals.append(planes)
        if cd is not None: cd_vals.append(cd)
        if f1 is not None: f1_vals.append(f1)

    planes_ranked = sorted(list(set(planes_vals)))
    cd_ranked = sorted(list(set(cd_vals)))
    f1_ranked = sorted(list(set(f1_vals)), reverse=True)

    def get_color_str(val, ranked_list, fmt):
        if val is None:
            return "-"
        str_val = f"{val:{fmt}}"
        try:
            rank = ranked_list.index(val)
            if rank == 0:
                return f"\\cellcolor{{red!25}}{str_val}"
            elif rank == 1:
                return f"\\cellcolor{{orange!25}}{str_val}"
            elif rank == 2:
                return f"\\cellcolor{{yellow!25}}{str_val}"
            else:
                return str_val
        except ValueError:
            return str_val

    for i, row in enumerate(rows_data):
        planes_str = f"{row['planes']:.1f}" if row['planes'] is not None else "-"
        cd_str     = get_color_str(row['cd'], cd_ranked, ".2f")
        f1_str     = get_color_str(row['f1'], f1_ranked, ".2f")

        if row['group'] == "loo_first":
            print("\\midrule")
            print("\\textit{Leave-one-out:} & & & \\\\")
        elif row['group'] == "iso_first":
            print("\\midrule")
            print("\\textit{Isolation:} & & & \\\\")
        elif row['group'] == "all_out":
            print("\\midrule")
            print("\\textit{Leave-all-out:} & & & \\\\")
        elif row['group'] == "full":
            print("\\midrule")

        print(f"{row['label']} & {planes_str} & {cd_str} & {f1_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main_ablation_tnt():
    """
    Generates the LaTeX ablation table comparing all ablation variants vs. the full model on TnT Barn.
    """
    import re

    BASE_DIR = '/Users/mchu/Documents/TUD/Thesis/PlanarSplatting'

    VARIANTS = [
        ("Replace normal source (to Metric3Dv2 \\citep{hu_metric3dv2_2024})", "evaluation/eval_tnt/ablation/Normalswap", "Ablation_tnt/Normalswap", "normal"),
        ("w/o Mesh post-processing", "evaluation/eval_tnt/ablation/No1mesh", "Ablation_tnt/No1mesh", "loo_first"),
        ("w/o Mask-Guided Densification \\& Pruning", "evaluation/eval_tnt/ablation/Nosplit", "Ablation_tnt/Nosplit", "loo"),
        ("w/o Final Mask-Guided Trim", "evaluation/eval_tnt/ablation/Notrim", "Ablation_tnt/Notrim", "loo"),
        ("Only Mesh post-processing", "evaluation/eval_tnt/ablation/Only1mesh", "Ablation_tnt/Only1mesh", "iso_first"),
        ("Only Mask-Guided Densification \\& Pruning", "evaluation/eval_tnt/ablation/Onlysplit", "Ablation_tnt/Onlysplit", "iso"),
        ("Only Final Mask-Guided Trim", "evaluation/eval_tnt/ablation/Onlytrim", "Ablation_tnt/Onlytrim", "iso"),
        ("All 3 modules disabled", "evaluation/eval_tnt/ablation/Allnone", "Ablation_tnt/Allnone", "all_out"),
        ("Full model", "evaluation/eval_tnt/APS", "AdaptivePS/TnT", "full"),
    ]

    def find_latest_run(scene_dir):
        try:
            entries = sorted([e for e in os.listdir(scene_dir)
                              if os.path.isdir(os.path.join(scene_dir, e))])
            return os.path.join(scene_dir, entries[-1]) if entries else None
        except Exception:
            return None

    def get_final_planar_instances(log_file):
        pattern = re.compile(r"number of planar instances = (\d+)")
        final = None
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        final = int(m.group(1))
        except Exception:
            pass
        return final

    def get_tnt_metrics(eval_dir_abs):
        chamfer_path = os.path.join(eval_dir_abs, "Barn.chamfer.txt")
        prf_path = os.path.join(eval_dir_abs, "Barn.prf_tau_plotstr.txt")
        
        overall = None
        f1 = None
        
        if os.path.exists(chamfer_path):
            try:
                data = np.loadtxt(chamfer_path)
                overall = data[2] * 100.0  # meters to cm
            except Exception:
                pass
                
        if os.path.exists(prf_path):
            try:
                data = np.loadtxt(prf_path)
                f1 = data[2]
            except Exception:
                pass
                
        return overall, f1

    # --- print table ---
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\footnotesize")
    print("\\caption{Ablation study on \\ac{TnT} \\textit{Barn} (taking mean values). \"Red\", \"Orange\" and \"Yellow\" denote the top 1-3 results.}")
    print("\\label{tab:ablation_tnt}")
    print("\\begin{tabular}{l ccc}")
    print("\\toprule")
    print("Model Setting & Planes no. $\\downarrow$ & CD (cm)$\\downarrow$ & F1-score @ 1cm $\\uparrow$ \\\\")
    print("\\midrule")

    rows_data = []
    planes_vals = []
    cd_vals = []
    f1_vals = []

    for i, (label, eval_rel, runs_rel, group) in enumerate(VARIANTS):
        eval_abs = os.path.join(BASE_DIR, eval_rel)
        runs_abs = os.path.join(BASE_DIR, runs_rel)

        # only one scene in TNT Barn so mean planes = planes of Barn
        barn_runs_dir = os.path.join(runs_abs, "Barn_APS")
        planes = None
        latest = find_latest_run(barn_runs_dir)
        if latest is not None:
            log = os.path.join(latest, 'train.log')
            planes = get_final_planar_instances(log)

        cd, f1 = get_tnt_metrics(eval_abs)

        rows_data.append({
            'label': label,
            'planes': planes,
            'cd': cd,
            'f1': f1,
            'group': group
        })

        if planes is not None: planes_vals.append(planes)
        if cd is not None: cd_vals.append(cd)
        if f1 is not None: f1_vals.append(f1)

    planes_ranked = sorted(list(set(planes_vals)))
    cd_ranked = sorted(list(set(cd_vals)))
    f1_ranked = sorted(list(set(f1_vals)), reverse=True)

    def get_color_str(val, ranked_list, fmt):
        if val is None:
            return "-"
        str_val = f"{val:{fmt}}"
        try:
            rank = ranked_list.index(val)
            if rank == 0:
                return f"\\cellcolor{{red!25}}{str_val}"
            elif rank == 1:
                return f"\\cellcolor{{orange!25}}{str_val}"
            elif rank == 2:
                return f"\\cellcolor{{yellow!25}}{str_val}"
            else:
                return str_val
        except ValueError:
            return str_val

    for i, row in enumerate(rows_data):
        planes_str = f"{row['planes']:.0f}" if row['planes'] is not None else "-"
        cd_str     = get_color_str(row['cd'], cd_ranked, ".2f")
        f1_str     = get_color_str(row['f1'], f1_ranked, ".4f")

        if row['group'] == "loo_first":
            print("\\midrule")
            print("\\textit{Leave-one-out:} & & & \\\\")
        elif row['group'] == "iso_first":
            print("\\midrule")
            print("\\textit{Isolation:} & & & \\\\")
        elif row['group'] == "all_out":
            print("\\midrule")
            print("\\textit{Leave-all-out:} & & & \\\\")
        elif row['group'] == "full":
            print("\\midrule")

        print(f"{row['label']} & {planes_str} & {cd_str} & {f1_str} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    
    # main_ablation()

    main_ablation_tnt()
