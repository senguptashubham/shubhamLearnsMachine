import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = 'results'

COLORS = {'mine': '#2a78d6', 'ref': '#e34948'}      # implementation -> hue
LINESTYLES = {'cpu': '-', 'cuda': '--'}              # device -> line style
LABELS = {'mine': 'MyConv2d', 'ref': 'nn.Conv2d'}

def latest_results_file():
    files = glob.glob(os.path.join(RESULTS_DIR, 'benchmark_*.csv'))
    return max(files, key=os.path.getmtime)

def load_results(path):
    df = pd.read_csv(path)
    return df[df['error'].isna()]

def plot(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    for device in sorted(df['device'].unique()):
        sub = df[df['device'] == device].sort_values('size')
        for impl in ['mine', 'ref']:
            ax.plot(
                sub['size'], sub[impl],
                color=COLORS[impl], linestyle=LINESTYLES[device],
                marker='o', markersize=5, linewidth=2,
                label=f'{LABELS[impl]} ({device})'
            )

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sorted(df['size'].unique()))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel('Spatial size (H = W)', color='#52514e')
    ax.set_ylabel('Mean time per forward pass (s)', color='#52514e')
    ax.set_title('MyConv2d vs nn.Conv2d — CPU vs GPU scaling', color='#0b0b0b', fontsize=13)

    ax.grid(True, which='both', color='#e1e0d9', linewidth=0.7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors='#898781')

    ax.legend(frameon=False, labelcolor='#52514e')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

if __name__ == '__main__':
    path = latest_results_file()
    print("Plotting", path)
    df = load_results(path)
    plot(df, out_path=os.path.join(RESULTS_DIR, 'benchmark_plot.png'))
