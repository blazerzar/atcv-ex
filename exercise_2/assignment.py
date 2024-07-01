from os import listdir, makedirs, path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ex2_utils import generate_responses_1
from ms_mode import find_mode, generate_responses_2
from ms_tracker import MSParams
from run_params import run_tracker

DATASET = 'vot2014'
SAVE_FIG = False
SAVE_RESULTS = False
SHOW_PLOTS = True
FIGS_DIR = 'figures'
RESULTS_DIR = 'results'


def main() -> None:
    plt.rcParams['figure.dpi'] = 150
    if SAVE_FIG and not path.exists(FIGS_DIR):
        makedirs(FIGS_DIR)
    if SAVE_RESULTS and not path.exists(RESULTS_DIR):
        makedirs(RESULTS_DIR)

    all_sequences = sorted(
        [n for n in listdir(DATASET) if path.isdir(path.join(DATASET, n))]
    )
    eval_sequences = [
        'basketball',
        'bicycle',
        'bolt',
        'fish1',
        'hand2',
        'tunnel',
    ]

    eval_mean_shift()
    print()

    eval_vot(all_sequences, eval_sequences)
    eval_mean_shift_custom()

    if SAVE_RESULTS:
        eval_parameters(all_sequences)

    eval_improvements(all_sequences, eval_sequences)


def eval_mean_shift():
    """Evaluate mean shift mode seeking convergence."""
    starting_points = np.array(
        [
            [29, 23, 15, 48, 89, 79],
            [88, 57, 16, 43, 58, 68],
        ]
    )
    responses = generate_responses_1()
    fig, ax = plt.subplots(1, 3, figsize=(5, 2))
    for a in ax:
        a.imshow(responses, cmap='gist_heat')
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_ylim(a.get_ylim()[::-1])
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

    print('unif  gauss large adapt')
    for i in range(starting_points.shape[1]):
        x = starting_points[:, i]
        x_unif = find_mode(x, responses, h=23, k='unif')
        x_gauss = find_mode(x, responses, h=23, k='gauss')
        x_large = find_mode(x, responses, h=63, k='unif')
        x_adapt = find_mode(x, responses, h=63, k='unif', adapt=True)
        print(
            f'{x_unif.shape[1]:^5} {x_gauss.shape[1]:^5} '
            f'{x_large.shape[1]:^5} {x_adapt.shape[1]:^5}'
        )

        for a in ax:
            a.plot(x_unif[0], x_unif[1], '.', markersize=1, color='violet')
        ax[0].plot(x_gauss[0], x_gauss[1], '.', markersize=1, color='lightblue')
        ax[1].plot(x_large[0], x_large[1], '.', markersize=1, color='lightblue')
        ax[2].plot(x_adapt[0], x_adapt[1], '.', markersize=1, color='lightblue')
        for a in ax:
            a.plot(*x, 'wx', markersize=4)
        ax[0].text(x[0] - 4, x[1] + 4, f'{i + 1}', color='white', fontsize=6)

    if SAVE_FIG:
        fig.savefig(path.join(FIGS_DIR, 'mean_shift.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def eval_vot(sequences: list[str], eval_sequences: list[str]):
    """Evaluate tracker performance on VOT sequences. The worse perfoming
    sequences were selected and fail cases were analyzed."""
    all_failures = 0
    for sequence in sequences:
        parameters = MSParams()
        fps, failures = run_tracker(sequence, parameters)
        all_failures += len(failures)
        if sequence in eval_sequences:
            print(f'{sequence}: {fps:.2f} fps, {len(failures)} failures')

            for i, failure in enumerate(failures):
                frame = cv2.rectangle(failure.image, *failure.truth, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, *failure.detect, (255, 0, 0), 2)
                plt.imsave(path.join(FIGS_DIR, f'{sequence}_{i}.png'), frame)

    print('Total failures:', all_failures)


def eval_mean_shift_custom():
    """Evaluate mean shift mode seeking convergence on a custom function."""
    angles = np.linspace(0, 2 * np.pi, 9)
    radii = np.array([[0] + [15] * 8])
    starting_points = np.vstack(
        [
            50 + np.cos(angles) * radii,
            50 + np.sin(angles) * radii,
        ]
    )
    responses = generate_responses_2()

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(responses, cmap='gist_heat')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

    for i in range(starting_points.shape[1]):
        x = starting_points[:, i]
        x_unif = find_mode(x, responses, h=23, k='unif')
        x_adapt = find_mode(x, responses, h=61, k='unif', adapt=True)

        ax.plot(x_unif[0], x_unif[1], '.', markersize=1, color='violet')
        ax.plot(x_adapt[0], x_adapt[1], '.', markersize=1, color='lightblue')
        ax.plot(*x, 'wx', markersize=4)

    if SAVE_FIG:
        fig.savefig(path.join(FIGS_DIR, 'griewank.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def eval_parameters(sequences: list[str]):
    """Evaluate tracker performance with different parameters."""
    bins_values = [4, 8, 12, 16]
    iter_values = [5, 10, 15, 20]
    alpha_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    eps_values = [0.25, 0.5, 0.75, 1.0]
    k_values = [0.25, 0.5, 0.75, 1.0, 1.25]
    bins = [MSParams(bins=b) for b in bins_values]
    max_iter = [MSParams(max_iter=i) for i in iter_values]
    alpha = [MSParams(alpha=a) for a in alpha_values]
    eps = [MSParams(eps=e) for e in eps_values]
    k_size = [MSParams(k_size=k) for k in k_values]

    eval_parameter(sequences, bins, bins_values, 'bins')
    eval_parameter(sequences, max_iter, iter_values, 'max_iter')
    eval_parameter(sequences, alpha, alpha_values, 'alpha')
    eval_parameter(sequences, eps, eps_values, 'eps')
    eval_parameter(sequences, k_size, k_values, 'k_size')


def eval_improvements(sequences: list[str], eval_sequences: list[str]):
    """Evaluate the tracker performance with different improvements."""
    for params, name in [
        (MSParams(bg=True), 'bg'),
        (MSParams(color_space='HSV'), 'HSV'),
        (MSParams(color_space='YCrCb'), 'YCrCb'),
        (MSParams(color_space='Lab'), 'Lab'),
    ]:
        print('\nImprovement:', name)
        total_failures = 0
        for sequence in sequences:
            fps, failures = run_tracker(sequence, params)
            total_failures += len(failures)
            if sequence in eval_sequences:
                print(f'{sequence}: {fps:.2f} fps, {len(failures)} failures')
        print('Total failures:', total_failures)


def eval_parameter(
    sequences: list[str], parameters: list[MSParams], values: list, name: str
):
    """Evaluate the tracker performance with different parameters."""
    results = pd.DataFrame(
        {'sequence': sequences} | {f'{name}={value}': 0 for value in values}
    )

    total_failures = []
    for j, params in enumerate(parameters):
        total_failures.append(0)

        for i, sequence in enumerate(sequences):
            _, failures = run_tracker(sequence, params)
            total_failures[-1] += len(failures)
            results.iloc[i, j + 1] = len(failures)

    results.loc[len(sequences)] = ('sum', *total_failures)  # type: ignore
    results.to_csv(path.join(RESULTS_DIR, f'{name}.csv'), index=False)


if __name__ == '__main__':
    main()
