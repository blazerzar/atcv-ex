from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ex4_utils import kalman_step
from kalman_filter import nca_model, ncv_model, rw_model
from particle_filter import PFParams
from run_params import eval_vot

LINE_WIDTH = 3.220
FIGS_DIR = 'figures'
SAVE_FIG = False
SHOW_PLOTS = False


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

    eval_kalman_filter()
    eval_curves()
    eval_tracker()
    eval_motion_models()
    eval_num_particles()


def eval_kalman_filter() -> None:
    """Evaluate the Kalman filter on a spiral path."""
    _, ax = plt.subplots(3, 5, figsize=(8, 5))
    for i, (q, r) in enumerate([(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]):
        eval_curve_motion_models(ax, i, q, r, spiral_path(40))
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'kalman_spiral.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def eval_curves() -> None:
    """Evaluate the Kalman filter on different curves."""
    _, ax = plt.subplots(3, 3, figsize=(4.5, 5))
    for i, (q, r) in enumerate([(100, 1), (1, 1), (1, 100)]):
        eval_curve_motion_models(ax, i, q, r, hilbert_curve())
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'kalman_hilbert.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    _, ax = plt.subplots(3, 3, figsize=(10, 4))
    for i, (q, r) in enumerate([(100, 1), (1, 1), (1, 100)]):
        eval_curve_motion_models(ax, i, q, r, sawtooth_path(10))
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'kalman_sawtooth.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def eval_curve_motion_models(
    ax, column: int, q: float, r: float, path: tuple[NDArray, NDArray]
) -> None:
    ax[0, column].set_title(f'RW: q={q}, r={r}', fontsize=8)
    ax[1, column].set_title(f'NCV: q={q}, r={r}', fontsize=8)
    ax[2, column].set_title(f'NCA: q={q}, r={r}', fontsize=8)
    for i in range(3):
        ax[i, column].tick_params(axis='both', which='major', labelsize=6)
    run_kalman_filter(path, rw_model, q, r, ax[0, column])
    run_kalman_filter(path, ncv_model, q, r, ax[1, column])
    run_kalman_filter(path, nca_model, q, r, ax[2, column])


def eval_tracker() -> None:
    """Evaluate the particle filter tracker on the VOT dataset."""
    print(
        'NCV results: {} failures, {:.2f} overlap, {:.1f} fps'.format(
            *eval_vot(PFParams(q=0.6))
        )
    )


def eval_motion_models() -> None:
    """Evaluate the particle filter on different motion models."""
    if not path.exists(f'rw_results.csv'):
        eval_motion_parameter('rw')
    if not path.exists(f'ncv_results.csv'):
        eval_motion_parameter('ncv')
    plot_motion_parameter()

    print(
        'NCA results: {} failures, {:.2f} overlap, {:.1f} fps'.format(
            *eval_vot(PFParams(motion_model='nca', q=1e-6))
        )
    )


def eval_motion_parameter(motion_model: str) -> None:
    qs = np.linspace(0.1, 2, 20)
    results = pd.DataFrame(columns=['q', 'failures', 'overlap', 'fps'])
    for i, q in enumerate(qs):
        params = PFParams(motion_model=motion_model, q=q)
        try:
            results.loc[i] = q, *eval_vot(params)  # type: ignore
        except ValueError:
            pass

    results.to_csv(f'{motion_model}_results.csv', index=False)


def plot_motion_parameter() -> None:
    rw = pd.read_csv('rw_results.csv')
    ncv = pd.read_csv('ncv_results.csv')

    plt.figure(figsize=(LINE_WIDTH, 0.6 * LINE_WIDTH))
    plt.plot(rw['q'], rw['failures'], color='black', label='RW')
    plt.plot(ncv['q'], ncv['failures'], color='black', linestyle='--', label='NCV')
    plt.xlim(0.1, 2)
    plt.xticks([0.1, 0.5, 1.0, 1.5, 2.0])
    plt.xlabel('$q$')
    plt.ylabel('Failures')
    plt.tight_layout()
    frame = plt.legend(borderpad=0.1).get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor('black')
    frame.set_boxstyle('square')  # type: ignore

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, f'motion_parameter.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def eval_num_particles() -> None:
    """Evaluate the particle filter on different number of particles."""
    if not path.exists(f'particles_results.csv'):
        eval_particles_parameter()
    plot_num_particles()


def eval_particles_parameter() -> None:
    num_particles = [50, 100, 150, 200, 250, 300]
    results = pd.DataFrame(columns=['num_particles', 'failures', 'overlap', 'fps'])
    for i, particles in enumerate(num_particles):
        params = PFParams(q=0.6, num_particles=particles)
        results.loc[i] = particles, *eval_vot(params)  # type: ignore
        print(results.loc[i].to_numpy())

    results.to_csv(f'particles_results.csv', index=False)


def plot_num_particles() -> None:
    results = pd.read_csv('particles_results.csv')

    plt.figure(figsize=(1.1 * LINE_WIDTH, 0.6 * LINE_WIDTH))
    plt.plot(
        results['num_particles'], results['failures'], color='black', label='Fails'
    )
    plt.xlim(20, 300)
    plt.xticks([20, 100, 200, 300])
    plt.xlabel('Particles')
    plt.ylabel('Failures')
    plt.tight_layout()

    right_axis = plt.gca().twinx()
    right_axis.plot(  # type: ignore
        results['num_particles'],
        results['fps'],
        color='black',
        label='FPS',
        linestyle='--',
    )
    right_axis.set_ylabel('FPS')
    plt.tight_layout()

    labels = ['Fails', 'FPS']
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=0.8),
        plt.Line2D([0], [0], color='black', linewidth=0.8, linestyle='--'),
    ]
    frame = plt.legend(handles, labels, borderpad=0.1).get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor('black')
    frame.set_boxstyle('square')  # type: ignore

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, f'num_particles.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def spiral_path(N: int) -> tuple[NDArray, NDArray]:
    v = np.linspace(5 * np.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    return x, y


def hilbert_curve() -> tuple[NDArray, NDArray]:
    part = np.array(
        [
            [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3],
            [0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0],
        ]
    )
    curve = np.concatenate(
        [
            part[::-1],
            part + np.array([[0], [4]]),
            part + np.array([[4], [4]]),
            np.array([[7], [3]]) - part[::-1],
        ],
        axis=1,
    )

    return curve[0], curve[1]


def sawtooth_path(size: int) -> tuple[NDArray, NDArray]:
    x = np.linspace(0, 10, 4 * size + 1)
    y = np.array([0, 20 / size, 0, 10 / size] * size + [0])
    return x, y


def run_kalman_filter(path: tuple[NDArray, NDArray], model, q: float, r: float, ax):
    x, y = path
    predictions = np.zeros((x.size, 2))
    predictions[0] = x[0], y[0]

    A, C, Q, R = model(q, r)
    P_k = np.eye(A.shape[0], dtype=np.float32)
    if A.shape[0] > 2:
        P_k[2:, 2:] *= 2

    state = np.zeros((len(A), 1), dtype=np.float32)
    state[:2, 0] = x[0], y[0]

    for j in range(1, x.size):
        measurement = np.array([x[j], y[j]]).reshape(-1, 1)
        state, P_k, _, _ = kalman_step(A, C, Q, R, measurement, state, P_k)
        predictions[j] = state[:2].flatten()

    kwargs = {'linewidth': 0.8, 'markerfacecolor': 'none', 'markersize': 4}
    ax.plot(x, y, 'ko--', **kwargs)  # type: ignore
    ax.plot(predictions[:, 0], predictions[:, 1], 'bo-', **kwargs)  # type: ignore
    ax.axis('equal')


if __name__ == '__main__':
    main()
