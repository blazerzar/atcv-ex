from os import environ
from time import time

import cv2
import numpy as np
from numpy.typing import NDArray

from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

parent = object
if __name__ != 'dcf_tracker':
    from toolkit.utils.tracker import Tracker  # type: ignore

    parent = Tracker  # type: ignore


class DCFTracker(parent):
    def __init__(self, sigma=None, lambda_=None, alpha=None, scaling=None, mosse=False):
        """Default parameters are read from environment variables DCF_SIGMA,
        DCF_LAMBDA, DCF_ALPHA, and DCF_SCALING. If environment variables are
        not set, fall back to default values.
        """
        sigma_default = float(environ.get('DCF_SIGMA', 1.75))
        lambda_default = float(environ.get('DCF_LAMBDA', 1e2))
        alpha_default = float(environ.get('DCF_ALPHA', 0.10))
        scaling_default = float(environ.get('DCF_SCALING', 1.00))

        self.sigma = sigma or sigma_default
        self.lambda_ = lambda_ or lambda_default
        self.alpha = alpha if alpha is not None else alpha_default
        self.scaling = scaling or scaling_default
        self.mosse = mosse

        self.initialize_times = []
        self.track_times = []

    def initialize(self, image: NDArray, region: list) -> None:
        """Initialize the tracker with the first frame and the region to track.

        Parameters:
            - image: Image to initialize the tracker
            - region: Initial region to track [left, top, width, height]
        """
        start = time()
        image = self._preprocess(image)

        left, top, width, height = region
        center_y, center_x = top + height / 2, left + width / 2
        self.x_k = np.array([center_x, center_y])
        self.box = np.array([width, height])

        # Desired impulse response
        w, h = self.scaling * width, self.scaling * height
        response = create_gauss_peak((w, h), self.sigma)
        self.g_hat = np.fft.fft2(response)
        self.search = np.array(response.shape[::-1])

        # Weighted image patch
        patch, _ = get_patch(image, self.x_k, self.search)
        self.hanning = create_cosine_window(self.search)
        f_hat = np.fft.fft2(patch * self.hanning)

        a, b = compute_correlation_filter(f_hat, self.g_hat, self.lambda_)
        self.filter = a / b
        if self.mosse:
            self.a, self.b = a, b

        self.initialize_times.append(time() - start)

    def track(self, image: NDArray) -> list[float]:
        """Track the region in the next frame."""
        start = time()
        image = self._preprocess(image)
        patch, mask = get_patch(image, self.x_k, self.search)
        f_hat = np.fft.fft2(patch * mask * self.hanning)

        response = np.fft.ifft2(f_hat * self.filter)
        y, x = np.unravel_index(np.argmax(response), response.shape)
        x -= (x > response.shape[1] / 2) * response.shape[1]
        y -= (y > response.shape[0] / 2) * response.shape[0]
        self.x_k += [x, y]

        # Filter update
        patch, mask = get_patch(image, self.x_k, self.search)
        f_hat = np.fft.fft2(patch * mask * self.hanning)

        a, b = compute_correlation_filter(f_hat, self.g_hat, self.lambda_)
        if self.mosse:
            self.a = (1 - self.alpha) * self.a + self.alpha * a
            self.b = (1 - self.alpha) * self.b + self.alpha * b
            self.filter = self.a / self.b
        else:
            self.filter = (1 - self.alpha) * self.filter + self.alpha * a / b

        left, top = self.x_k - self.box // 2
        self.track_times.append(time() - start)
        return [left, top, *self.box]

    def name(self) -> str:
        return 'dcf'

    def _preprocess(self, image: NDArray) -> NDArray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, std = np.mean(image), np.std(image)
        return (image - mean) / std

    def get_performance(self) -> tuple[float, float]:
        return np.mean(self.initialize_times), np.mean(self.track_times)


def compute_correlation_filter(
    patch: NDArray, response: NDArray, l: float
) -> tuple[NDArray, NDArray]:
    """Compute the correlation filter for a given patch that produces the
    desired response when correlated with the patch.

    Parameters:
        - patch (f): Search region patch in the Fourier domain
        - response (g): Desired impulse response in the Fourier domain
        - l: Regularization parameter lambda

    Returns:
        - filter (a, b): Conjugated filter in the Fourier domain
    """
    if patch.shape != response.shape:
        print(patch.shape, response.shape)
        raise ValueError('Patch and response must have the same shape')

    f_conjugate = np.conj(patch)
    a = response * f_conjugate
    b = patch * f_conjugate + l
    return a, b
