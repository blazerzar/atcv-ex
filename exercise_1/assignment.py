from os import makedirs, path
from time import time
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ex1_utils import rotate_image, show_flow
from of_methods import hornschunck, lucaskanade, lucaskanade_pyramids

SAVE_FIG = False
SHOW_PLOTS = True
FIGS_DIR = 'figures'


def main():
    plt.rcParams['figure.dpi'] = 150
    if SAVE_FIG and not path.exists(FIGS_DIR):
        makedirs(FIGS_DIR)

    random_noise_evaluation()
    methods_evaluation()
    harris_evaluation()
    lk_params_evaluation()
    hs_params_evaluation()
    time_evaluation()
    pyramidal_evaluation()


def random_noise_evaluation() -> None:
    """Evaluate the optical flow methods on a pair of random noise images,
    where the second image is a rotated version of the first one."""
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = rotate_image(im1, -1)

    image_evaluation(im1, im2, cmap='viridis')
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'random_noise.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def methods_evaluation() -> None:
    """Evaluate the optical flow methods on three pairs of images."""
    image_evaluation('collision/00000170.jpg', 'collision/00000171.jpg')
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'collision.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    image_evaluation('disparity/cporta_left.png', 'disparity/cporta_right.png')
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'cporta.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    image_evaluation('lab2/027.jpg', 'lab2/028.jpg')
    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'lab.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def image_evaluation(
    im1: NDArray | str, im2: NDArray | str, cmap: str = 'gray'
) -> None:
    """Compute the optical flow between two images using the Lucas-Kanade and
    Horn-Schunck methods. The results are shown in a 2x3 grid of subplots."""
    if isinstance(im1, str):
        im1 = plt.imread(im1) / 255.0
    if isinstance(im2, str):
        im2 = plt.imread(im2) / 255.0
    U_lk, V_lk = lucaskanade(im1, im2, 3)
    U_hs, V_hs = hornschunck(im1, im2, 1000, 0.5)
    h, w = im1.shape

    _, ax = plt.subplots(2, 3, figsize=(6, 4 * h / w))
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05 * h / w, 0.05)

    ax[0, 0].imshow(im1, cmap=cmap)
    ax[1, 0].imshow(im2, cmap=cmap)
    show_flow(U_lk, V_lk, ax[0, 1], type='angle')
    show_flow(U_lk, V_lk, ax[1, 1], type='field', set_aspect=True)
    show_flow(U_hs, V_hs, ax[0, 2], type='angle')
    show_flow(U_hs, V_hs, ax[1, 2], type='field', set_aspect=True)


def harris_evaluation() -> None:
    """Evalute the Lucas-Kanade method with the Harris response improvement.
    Its value is used to detect areas of lower reliability."""
    im1 = plt.imread('disparity/cporta_left.png') / 255.0
    im2 = plt.imread('disparity/cporta_right.png') / 255.0
    U_lk, V_lk = lucaskanade(im1, im2, 3, harris=True)
    h, w = im1.shape

    _, ax = plt.subplots(1, 2, figsize=(6, 3 * h / w))
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05 * h / w, 0.05)

    show_flow(U_lk, V_lk, ax[0], type='angle')
    show_flow(U_lk, V_lk, ax[1], type='field', set_aspect=True)

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'harris.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def lk_params_evaluation() -> None:
    """Evalute how Lucas-Kanade's parameters affect the optical flow."""
    im1 = plt.imread('collision/00000170.jpg') / 255.0
    im2 = plt.imread('collision/00000171.jpg') / 255.0
    flows = [lucaskanade(im1, im2, N) for N in (3, 9, 15)]
    params_plots(flows, *im1.shape)

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'lk_params.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def hs_params_evaluation() -> None:
    """Evalute how Horn-Schunck's parameters affect the optical flow."""
    im1 = plt.imread('collision/00000170.jpg') / 255.0
    im2 = plt.imread('collision/00000171.jpg') / 255.0
    flows = [hornschunck(im1, im2, 1000, lmbd) for lmbd in (0.001, 1.0, 100.0)]
    params_plots(flows, *im1.shape)

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'hs_params.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def params_plots(flows: list[tuple[NDArray, NDArray]], h: int, w: int) -> None:
    """Plot the optical flow results for three different parameter values
    in a 2x3 grid of subplots."""
    _, ax = plt.subplots(2, 3, figsize=(6, 4 * h / w))
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05 * h / w, 0.05)

    show_flow(flows[0][0], flows[0][1], ax[0, 0], type='angle')
    show_flow(flows[0][0], flows[0][1], ax[1, 0], type='field', set_aspect=True)
    show_flow(flows[1][0], flows[1][1], ax[0, 1], type='angle')
    show_flow(flows[1][0], flows[1][1], ax[1, 1], type='field', set_aspect=True)
    show_flow(flows[2][0], flows[2][1], ax[0, 2], type='angle')
    show_flow(flows[2][0], flows[2][1], ax[1, 2], type='field', set_aspect=True)


def time_evaluation() -> None:
    """Evaluate the time performance of the optical flow methods."""
    im1 = plt.imread('collision/00000170.jpg') / 255.0
    im2 = plt.imread('collision/00000171.jpg') / 255.0
    h, w = im1.shape

    lk_time = timeit(lambda: lucaskanade(im1, im2, 7), number=500)
    print(f'Lucas-Kanade: {lk_time * 2:.2f} ms')

    start, iters = time(), [0]
    for _ in range(3):
        U_hs, V_hs = hornschunck(im1, im2, 20000, 0.5, tol=1e-4, convergence=iters)
    hs_time = (time() - start) / 3
    print(f'Horn-Schunck: {hs_time:.2f} s ({iters[0]} iterations)')

    start = time()
    for _ in range(3):
        U_hs_lk, V_hs_lk = hornschunck(
            im1, im2, 20000, 0.5, tol=1e-4, lk_init=True, convergence=iters
        )
    hs_time = (time() - start) / 3
    print(
        f'Horn-Schunck with Lucas-Kanade initialization: {hs_time:.2f} s '
        f'({iters[0]} iterations)'
    )

    _, ax = plt.subplots(1, 2, figsize=(6, 3 * h / w))
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05 * h / w, 0.05)

    show_flow(U_hs, V_hs, ax[0], type='angle')
    show_flow(U_hs_lk, V_hs_lk, ax[1], type='angle')

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'lk_init.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def pyramidal_evaluation() -> None:
    """Evalute the pyramidal Lucas-Kanade method."""
    im1 = plt.imread('collision/00000170.jpg') / 255.0
    im2 = plt.imread('collision/00000171.jpg') / 255.0
    h, w = im1.shape

    U_lk, V_lk = lucaskanade(im1, im2, 9)
    U_lk_py_1, V_lk_py_1 = lucaskanade_pyramids(im1, im2, 9, min_size=4, repeat=1)
    U_lk_py_2, V_lk_py_2 = lucaskanade_pyramids(im1, im2, 9, min_size=4, repeat=5)

    _, ax = plt.subplots(1, 3, figsize=(6, 2 * h / w))
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05 * h / w, 0.05)

    show_flow(U_lk, V_lk, ax[0], type='angle')
    show_flow(U_lk_py_1, V_lk_py_1, ax[1], type='angle')
    show_flow(U_lk_py_2, V_lk_py_2, ax[2], type='angle')

    if SAVE_FIG:
        plt.savefig(path.join(FIGS_DIR, 'pyramidal.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
