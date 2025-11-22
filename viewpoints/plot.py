"""
Process and plot height estimation results.
"""
import numpy as np
import os
import matplotlib.pyplot as plt

def single_error_histogram(npz_file):
    """Plot histogram of error percentages from a .npz results file."""
    data = np.load(npz_file, allow_pickle=True)
    error_percentages = data['error_percentages']

    # Convert to array, replace None with NaN, then drop NaN
    error_percentages = np.array(error_percentages, dtype=object)
    error_percentages = error_percentages[error_percentages != None].astype(float)
    total = error_percentages.size
    if total == 0:
        print("No valid error percentages.")
        return

    bins = np.linspace(0, 100, 21)

    # Get counts then convert to proportions
    counts, edges = np.histogram(error_percentages, bins=bins)
    proportions = counts / total

    plt.figure(figsize=(10, 6))
    plt.bar(edges[:-1], proportions, width=np.diff(edges), align='edge',
            color='blue', alpha=0.7, edgecolor='black')
    plt.title('Height Estimation Error Percentages (Using 3 Orthogonal Views of Depth + Intrinsics + Extrinsics)')
    plt.xlabel('Error Percentage (%)')
    plt.ylabel('Proportion')
    plt.ylim(0, proportions.max()*1.15)
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Optional: print bin ranges and proportions
    for i in range(len(proportions)):
        print(f"Bin {edges[i]:.1f}–{edges[i+1]:.1f}%: {proportions[i]:.4f}")

def compare_two_error_histograms(npz_file1, npz_file2, label1='Method 1', label2='Method 2'):
    """Compare two histograms of error percentages from two .npz results files."""
    data1 = np.load(npz_file1, allow_pickle=True)
    error_percentages1 = data1['error_percentages']
    error_percentages1 = np.array(error_percentages1, dtype=object)
    error_percentages1 = error_percentages1[error_percentages1 != None].astype(float)

    data2 = np.load(npz_file2, allow_pickle=True)
    error_percentages2 = data2['error_percentages']
    error_percentages2 = np.array(error_percentages2, dtype=object)
    error_percentages2 = error_percentages2[error_percentages2 != None].astype(float)

    bins = np.linspace(0, 100, 21)

    counts1, edges = np.histogram(error_percentages1, bins=bins)
    proportions1 = counts1 / error_percentages1.size

    counts2, _ = np.histogram(error_percentages2, bins=bins)
    proportions2 = counts2 / error_percentages2.size

    width = np.diff(edges)[0] * 0.4  # bar width

    plt.figure(figsize=(10, 6))
    plt.bar(edges[:-1] - width/2, proportions1, width=width,
            label=label1, color='blue', alpha=0.7, edgecolor='black')
    plt.bar(edges[:-1] + width/2, proportions2, width=width,
            label=label2, color='orange', alpha=0.7, edgecolor='black')
    plt.title('Comparison of Height Estimation Error Percentages')
    plt.xlabel('Error Percentage (%)')
    plt.ylabel('Proportion')
    plt.ylim(0, max(proportions1.max(), proportions2.max())*1.15)
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # npz_file = "/Users/maxlyu/Documents/nutritionverse-3d-dataset-manual/height_estimation_results_0_105.npz"
    # npz_file = "/Users/maxlyu/Documents/nutritionverse-3d-dataset-manual/height_estimation_extrinsic_fixbug_0_105.npz"
    # single_error_histogram(npz_file)

    npz_file_1 = "/Users/maxlyu/Documents/nutritionverse-3d-dataset-manual/height_estimation_results_0_105.npz"
    npz_file_2 = "/Users/maxlyu/Documents/nutritionverse-3d-dataset-manual/height_estimation_extrinsic_fixbug_0_105.npz"
    compare_two_error_histograms(npz_file_1, npz_file_2, label1='Intrinsic', label2='Extrinsic')


if __name__ == "__main__":
    main()
