import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ex2_utils import generate_responses_1, get_patch


def mean_shift(x_k: NDArray, weights: NDArray, kernel: NDArray) -> NDArray:
    """Perform one iteration of the mean shift algorithm from the current
    position x_k = (x, y) on the weights image. The image is additionally
    weighted by the kernel.
    """
    if weights.shape != kernel.shape:
        raise ValueError('Weights and kernel must have the same shape')
    if weights.shape[0] % 2 == 0 or weights.shape[1] % 2 == 0:
        raise ValueError('Weights and kernel must have odd dimensions')

    h, w = weights.shape
    h_y, h_x = h // 2, w // 2
    xs = np.tile(np.arange(-h_x, h_x + 1), h)
    ys = np.repeat(np.arange(-h_y, h_y + 1), w)
    neighbours = np.stack((xs, ys))

    points = np.array([[h_x], [h_y]]) + neighbours
    n_weights = weights[points[1], points[0]] * kernel[points[1], points[0]]
    n_sum = np.sum(n_weights)

    if n_sum < 1e-10:
        return x_k

    mean_shift_vector = np.sum(neighbours * n_weights, axis=1) / n_sum
    return x_k + mean_shift_vector


def find_mode(x_0, weights, *, h=11, k='unif', adapt=False):
    """Converge the mean shift algorithm from the initial position x_0 on the
    weights image.

    Parameters:
        - x_0: initial position (x, y)
        - weights: image with shape (height, width)
        - h: bandwidth size
        - k: kernel type ('unif' or 'gauss')
        - adapt: adapt bandwidth size over iterations

    Returns:
        - x: trajectory of the mean shift algorithm with shape (2, n)
    """
    if h % 2 == 0:
        raise ValueError('Bandwidth size must be odd')

    x = [x_0]
    while True:
        # Create kernel and mask it
        radius = h // 2
        coords = np.arange(-radius, radius + 1)
        xs, ys = np.meshgrid(coords, coords)
        if k == 'unif':
            kernel = xs**2 + ys**2 <= radius**2
        elif k == 'gauss':
            kernel = np.exp(-(xs**2 + ys**2) / (radius / 3) ** 2)

        patch, mask = get_patch(weights, x[-1], (h, h))
        kernel[mask == 0] = 0

        x_new = mean_shift(x[-1], patch, kernel)
        if np.linalg.norm(x[-1] - x_new) < 0.4:
            return np.vstack(x).T
        x.append(x_new)

        if adapt:
            h = np.maximum(h * 0.8, 19).astype(int)
            h += 1 - h % 2


def generate_responses_2() -> NDArray:
    """Create a 100x100 image of the third order Griewank function."""
    x_3 = 12
    xs = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    x_1, x_2 = np.meshgrid(xs * 0.8, xs)

    return (
        1
        + 1 / 4000 * x_1**2
        + 1 / 4000 * x_2**2
        + 1 / 4000 * x_3**2
        - np.cos(x_1) * np.cos(x_2 * np.sqrt(2) / 2) * np.cos(x_3 * np.sqrt(3) / 3)
    )


def main() -> None:
    responses = generate_responses_1()
    # responses = generate_responses_2()

    fig, ax = plt.subplots()
    ax.imshow(responses, cmap='hot')

    def onclick(event):
        x = np.array([event.xdata, event.ydata]).round().astype(int)
        x_unif = find_mode(x, responses, h=23, k='unif')
        x_gauss = find_mode(x, responses, h=23, k='gauss')

        plt.plot(*x, 'wx')
        plt.plot(x_unif[0], x_unif[1], '.', markersize=1, color='violet')
        plt.plot(x_gauss[0], x_gauss[1], '.', markersize=1, color='lightblue')

        plt.ylim(0, responses.shape[0] - 1)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.ylim(0, responses.shape[0] - 1)
    plt.show()


if __name__ == '__main__':
    main()
