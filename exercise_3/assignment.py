from os import environ, listdir, makedirs, path
from subprocess import check_output
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore

from dcf_tracker import DCFTracker
from sequence_utils import VOTSequence

DATASET_DIR = path.join('workspace', 'sequences')
LINE_WIDTH = 3.220
FIGS_DIR = 'figures'
SAVE_FIG = False
SHOW_PLOTS = True


def main() -> None:
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

    evaluate_sigmas()
    evaluate_alphas()
    evaluate_scaling()
    evaluate_time()


def evaluate_sigmas() -> None:
    sigmas = np.linspace(0.5, 5, 19)
    results = np.zeros((len(sigmas), 3))
    for i, sigma in enumerate(tqdm(sigmas, ncols=60)):
        results[i] = evaluate_parameters(sigma, 1e2, 0.26, 1.0)
    output_results(results, sigmas, 'sigma', r'$\sigma$')


def evaluate_alphas() -> None:
    alphas = np.linspace(0.0, 0.4, 21)
    results = np.zeros((len(alphas), 3))
    for i, alpha in enumerate(tqdm(alphas, ncols=60)):
        results[i] = evaluate_parameters(1.75, 1e2, alpha, 1.0)
    output_results(results, alphas, 'alpha', r'$\alpha$')


def evaluate_scaling() -> None:
    scales = np.linspace(1.0, 1.5, 11)
    results = np.zeros((len(scales), 3))
    for i, scale in enumerate(tqdm(scales, ncols=60)):
        results[i] = evaluate_parameters(1.75, 1e2, 0.26, scale)
    output_results(results, scales, 'scaling', 'Scaling factor')


def evaluate_parameters(
    sigma: float, lambda_: float, alpha: float, scaling: float
) -> tuple[float, int, float]:
    """Evaluate the DCF tracker with the pytracking-lite toolkit."""
    environ['DCF_SIGMA'] = str(sigma)
    environ['DCF_LAMBDA'] = str(lambda_)
    environ['DCF_ALPHA'] = str(alpha)
    environ['DCF_SCALING'] = str(scaling)

    check_output(['bash', 'evaluate.sh'])
    results = check_output(['bash', 'measures.sh']).decode().splitlines()
    iou = float(results[3].split(': ')[1])
    fails = int(float(results[4].split(': ')[1]))
    fps = float(results[5].split(': ')[1][:-4])
    return iou, fails, fps


def evaluate_time() -> None:
    sequences = sorted(
        [
            name
            for name in listdir(DATASET_DIR)
            if path.isdir(path.join(DATASET_DIR, name))
        ]
    )

    results = np.array([measure_time(sequence) for sequence in sequences])
    results[:, 1:] *= 1000

    print('\nsequence   init   track   init - track')
    diff_sort = np.argsort(results[:, 1] - results[:, 2])
    for i in diff_sort:
        init, track = results[i, 1:]
        diff = init - track
        print(f'{sequences[i]:10} {init:.2f} {track:.2f}, {diff:+.2f}')

    average_results = np.mean(results, axis=0)
    print(f'Average FPS: {average_results[0]:.0f}')
    print(f'Average initialization time: {average_results[1]:.4f} ms')
    print(f'Average tracking time: {average_results[2]:.4f} ms')

    plt.figure(figsize=(LINE_WIDTH, 0.6 * LINE_WIDTH))
    plt.plot(results[:, 0], color='black')
    plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=5)
    plt.ylabel('FPS')
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'time.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def measure_time(sequence_title: str) -> tuple:
    """Return frames per second, average initialize time and average track
    time for the given sequence."""
    sequence = VOTSequence(DATASET_DIR, sequence_title)
    init_frame = 0
    tracker = DCFTracker()

    time_all = 0.0
    frame_idx = 0
    while frame_idx < sequence.length() and frame_idx < len(sequence.gt):
        img = cv2.imread(sequence.frame(frame_idx))
        start = time()
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        if frame_idx == init_frame:
            tracker.initialize(img, gt_bb)
            predicted_bbox = gt_bb
        else:
            predicted_bbox = tracker.track(img)
        time_all += time() - start

        o = sequence.overlap(predicted_bbox, gt_bb)
        frame_idx += 1 + (o == 0) * 4
        if o == 0:
            init_frame = frame_idx

    return sequence.length() / time_all, *tracker.get_performance()


def output_results(results: NDArray, values: NDArray, name: str, xlabel: str) -> None:
    best = np.argmin(results[:, 1])
    print(
        f'Optimal {name}: {values[best]:.2f}, IoU: {results[best, 0]:.2f}, '
        f'Fails: {results[best, 1]:.0f}, FPS: {results[best, 2]:.0f}'
    )

    plt.figure(figsize=(LINE_WIDTH, 0.5 * LINE_WIDTH))
    plt.plot(values, results[:, 1], color='black')
    plt.ylabel('Fails')
    plt.xlabel(xlabel)
    plt.xlim(values[0], values[-1])
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, f'{name}.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
