import csv
import numpy as np
from pathlib import Path

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def read_methods_and_stats(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Detect methods by columns ending with "_error_pct"
        err_cols = [c for c in fieldnames if c.endswith("_error_pct")]
        methods = []
        for err_col in err_cols:
            method_col = err_col[: -len("_error_pct")]
            if method_col in fieldnames:
                methods.append((method_col, err_col))

        totals = 0
        success = {m: 0 for m, _ in methods}
        errors = {m: [] for m, _ in methods}

        for row in reader:
            totals += 1
            for mcol, ecol in methods:
                val = safe_float(row.get(mcol))
                if val is not None:
                    success[mcol] += 1
                e = safe_float(row.get(ecol))
                if e is not None:
                    errors[mcol].append(e)

    method_names = [m for m, _ in methods]
    return totals, method_names, success, errors, csv_path.parent

def print_summary(totals, method_names, success, errors):
    print(f"Total rows: {totals}")
    print("Per-method stats:")
    for m in method_names:
        s_cnt = success[m]
        s_rate = (s_cnt / totals) * 100.0 if totals > 0 else 0.0
        errs = np.array(errors[m], dtype=float) if len(errors[m]) else None
        if errs is not None and errs.size > 0:
            mean_err = float(np.mean(errs))
            median_err = float(np.median(errs))
            print(f"- {m}: success_rate={s_rate:.2f}%  mean_error_pct={mean_err:.3f}  median_error_pct={median_err:.3f}  (n_err={errs.size})")
        else:
            print(f"- {m}: success_rate={s_rate:.2f}%  mean_error_pct=N/A  median_error_pct=N/A  (n_err=0)")

def plot_stats(method_names, success, totals, errors, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Success rate bar chart
    rates = [(success[m] / totals) * 100.0 if totals > 0 else 0.0 for m in method_names]
    plt.figure(figsize=(max(6, 0.6*len(method_names)), 4))
    x = np.arange(len(method_names))
    bars = plt.bar(x, rates, color="#4C78A8")
    plt.ylabel("Success rate (%)")
    plt.xticks(x, method_names, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.title("Success rate per method")
    # optional labels
    for b, r in zip(bars, rates):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_per_method.png", dpi=200)
    plt.close()

    # Prepare error data per method (skip empty)
    err_data = [(m, np.array(errors[m], dtype=float)) for m in method_names if len(errors[m]) > 0]
    
    # NOTE: Remove the "sphere_ransac_mean" method from plots if present (error too large)
    err_data = [(m, arr) for m, arr in err_data if m != "sphere_ransac_mean"]

    # Skip if no error data
    
    if len(err_data) == 0:
        return

    labels = [m for m, arr in err_data]
    arrays = [arr for _, arr in err_data]

    # 2) Error boxplot (distribution per method)
    plt.figure(figsize=(max(6, 0.6*len(labels)), 4))
    plt.boxplot(arrays, labels=labels, showfliers=False)
    plt.ylabel("Error (%)")
    plt.title("Error distribution per method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "error_boxplot_per_method.png", dpi=200)
    plt.close()

    # 3) Median error bar chart (with IQR as error bars)
    med = np.array([np.median(a) for a in arrays])
    q1 = np.array([np.percentile(a, 25) for a in arrays])
    q3 = np.array([np.percentile(a, 75) for a in arrays])
    err_lower = med - q1
    err_upper = q3 - med

    plt.figure(figsize=(max(6, 0.6*len(labels)), 4))
    xx = np.arange(len(labels))
    plt.bar(xx, med, yerr=[err_lower, err_upper], capsize=4, color="#F58518")
    plt.ylabel("Median error (%)")
    plt.title("Median error per method (IQR error bars)")
    plt.xticks(xx, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "error_median_bar_per_method.png", dpi=200)
    plt.close()

def summarize_methods(csv_path):
    totals, method_names, success, errors, out_dir = read_methods_and_stats(csv_path)
    print_summary(totals, method_names, success, errors)
    plot_stats(method_names, success, totals, errors, out_dir)

if __name__ == "__main__":
    summarize_methods("/Users/maxlyu/Documents/nutritionverse-3d-dataset/total.csv")
