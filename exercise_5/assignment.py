import glob
import sys
from os import devnull, environ, listdir, makedirs, path, remove

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore

from performance_evaluation import evaluate_performance
from run_tracker import evaluate_tracker

DATASET_DIR = 'dataset'
MODEL_PATH = 'siamfc_net.pth'
RESULTS_DIR = 'results'

LINE_WIDTH = 3.220
FIGS_DIR = 'figures'
SAVE_FIG = False
SHOW_PLOTS = True


def main() -> None:
    if not path.exists(RESULTS_DIR):
        makedirs(RESULTS_DIR)

    if SAVE_FIG and not path.exists(FIGS_DIR):
        makedirs(FIGS_DIR)

    # Plots configuration
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Latin Modern Roman'
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 10

    eval_short_term()
    eval_long_term()
    eval_threshold()
    eval_num_samples()
    eval_gaussian()


def eval_short_term() -> None:
    """Evaluate the short-term tracker."""
    p, r, f = run_params(False)
    print(f'Short-term tracker: P={p:.2f}, R={r:.2f}, F1={f:.2f}')


def eval_long_term() -> None:
    """Evaluate the long-term tracker."""
    p, r, f = run_params(True)
    print(f'Long-term tracker: P={p:.2f}, R={r:.2f}, F1={f:.2f}')


def eval_threshold() -> None:
    """Evaluate the long-term tracker with different threshold values."""
    if not path.exists('thresholds.csv'):
        results = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F1'])
        for t in np.linspace(2, 8, 7):
            p, r, f = run_params(True, t)
            results.loc[len(results)] = t, p, r, f
        results.to_csv('thresholds.csv', index=False)
    else:
        results = pd.read_csv('thresholds.csv')

    plt.figure(figsize=(LINE_WIDTH, 0.6 * LINE_WIDTH))
    plt.plot(
        results['Threshold'],
        results['F1'],
        marker='s',
        color='black',
        markerfacecolor='white',
    )
    plt.xlabel('Threshold')
    plt.ylabel(r'$F_1$ score')
    plt.xticks(results['Threshold'])

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'thresholds.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.tight_layout()
        plt.show()
    plt.close()


def eval_num_samples() -> None:
    """Evaluate whether the number of samples affects the number of frames it
    takes the tracker to re-detect the target."""
    if not path.exists('num_samples.csv'):
        sequences = [seq for seq in listdir(DATASET_DIR) if seq != 'list.txt']
        results = pd.DataFrame(columns=['Samples', *sequences])

        for s in [5, 10, 15, 20, 25, 30]:
            redetections = run_redetections(s)
            print(f'Num samples: {s}, Redetections: {redetections}')
            results.loc[len(results)] = s, *redetections
        results.to_csv('num_samples.csv', index=False)
    else:
        results = pd.read_csv('num_samples.csv')

    # Remove sitcom column because there are no re-detections
    results = results.drop(columns='sitcom')

    plt.figure(figsize=(LINE_WIDTH, 0.6 * LINE_WIDTH))
    for i, s in enumerate(results['Samples']):
        plt.plot(
            results.columns[1:],
            results.loc[results['Samples'] == s].values[0][1:],
            color='black',
            alpha=0.2 + 0.1 * i,
        )
    plt.xticks(rotation=90)
    plt.ylabel('Avg. re-detects')

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'num_samples.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.tight_layout()
        plt.show()
    plt.close()


def eval_gaussian() -> None:
    """Evaluate the long-term tracker with the Gaussian sampling strategy."""
    p, r, f = run_params(True, sampling='gaussian')
    print(f'Gaussian sampling: P={p:.2f}, R={r:.2f}, F1={f:.2f}')


def run_params(
    long_term: bool, threshold: float = 4, sampling: str = 'uniform', samples: int = 20
) -> tuple[float, float, float]:
    """Run the long-term tracker with the given parameters and evaluate the
    results using precision, recall and F1 score."""
    stdout = sys.stdout
    sys.stdout = open(devnull, 'w')

    for file in glob.glob(path.join(RESULTS_DIR, '*')):
        remove(file)

    environ['DCF_LONG_TERM'] = '1' if long_term else ''
    environ['DCF_THRESHOLD'] = str(threshold)
    environ['DCF_SAMPLING'] = sampling
    environ['DCF_NUM_SAMPLES'] = str(samples)
    evaluate_tracker(DATASET_DIR, MODEL_PATH, RESULTS_DIR, False)
    p, r, f = evaluate_performance(DATASET_DIR, RESULTS_DIR, return_results=True)

    sys.stdout = stdout
    return p, r, f


def run_redetections(samples: int) -> list[float]:
    """Run the tracker with the given number of samples and return the number of
    frames it takes to re-detect the target."""
    stdout = sys.stdout
    sys.stdout = open(devnull, 'w')

    for file in glob.glob(path.join(RESULTS_DIR, '*')):
        remove(file)

    environ['DCF_LONG_TERM'] = '1'
    environ['DCF_NUM_SAMPLES'] = str(samples)
    redetections = evaluate_tracker(DATASET_DIR, MODEL_PATH, RESULTS_DIR, False, True)

    sys.stdout = stdout
    return [np.mean(r) for r in redetections]


if __name__ == '__main__':
    main()
