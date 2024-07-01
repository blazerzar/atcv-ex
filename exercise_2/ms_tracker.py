from dataclasses import dataclass

import cv2
import numpy as np

from ex2_utils import (
    Tracker,
    backproject_histogram,
    create_epanechnik_kernel,
    extract_histogram,
    get_patch,
)
from ms_mode import mean_shift


@dataclass
class MSParams:
    bins: int = 16
    max_iter: int = 20
    alpha: float = 0.0
    eps: float = 0.5
    k_size: float = 0.5
    bg: bool = False
    color_space: str = 'RGB'


class MeanShiftTracker(Tracker):
    def __init__(self, params):
        if params.color_space not in {'RGB', 'HSV', 'YCrCb', 'Lab'}:
            raise ValueError('Invalid color space')

        self.bins = params.bins
        self.max_iter = params.max_iter
        self.alpha = params.alpha
        self.eps = params.eps
        self.k_size = params.k_size
        self.bg = params.bg
        self.color_space = eval(f'cv2.COLOR_BGR2{params.color_space}')

    def initialize(self, image, region):
        """Initialize the tracker with the first frame and the region to track.

        Parameters:
            - image: first frame of the sequence with shape (height, width, 3)
            - region: initial region to track with shape (l, t, width, height)
        """
        image = cv2.cvtColor(image, self.color_space)

        left, top, width, height = region
        center_y, center_x = top + height / 2, left + width / 2
        self.center = np.array([width / 2, height / 2]).round().astype(int)
        self.x_k = np.array([center_x, center_y])

        # Kernels for computing weighted histograms
        self.kernel = create_epanechnik_kernel(int(width), int(height), self.k_size)
        self.bg_kernel = create_epanechnik_kernel(
            int(np.sqrt(3) * width), int(np.sqrt(3) * height), 2
        )

        # Remove kernel pixels from the background kernel
        y, x = np.nonzero(self.kernel)
        dy = (self.bg_kernel.shape[0] - self.kernel.shape[0]) // 2
        dx = (self.bg_kernel.shape[1] - self.kernel.shape[1]) // 2
        self.bg_kernel[y + dy, x + dx] = 0

        # Region size has to equal the kernel size (always odd)
        self.box = np.array(self.kernel.shape)[::-1]
        self.bg_box = np.array(self.bg_kernel.shape)[::-1]

        self.q = self.get_target_model(image)

    def get_target_model(self, image):
        """Compute the target model histogram at the current position."""
        patch, mask = get_patch(image, self.x_k, self.box)
        self.bg_weights = self.get_background_histogram(image)
        q = extract_histogram(patch, self.bins, self.kernel * mask)
        if self.bg:
            q *= self.bg_weights
        return q / q.sum()

    def get_background_histogram(self, image):
        """Compute histogram of the background using a patch with three
        times bigger area than the region to track.
        """
        patch, mask = get_patch(image, self.x_k, self.bg_box)
        o = extract_histogram(patch, self.bins, self.bg_kernel * mask)
        o /= o.sum()

        bg_weights = np.minimum(np.min(o[o > 0]) / (o + 1e-15), 1)
        return bg_weights

    def track(self, image):
        """Compute the new position of the region in the next frame.

        Parameters:
            - image: next frame of the sequence with shape (height, width, 3)

        Returns:
            - new position of the region with shape (l, t, width, height)
        """
        image = cv2.cvtColor(image, self.color_space)

        for _ in range(self.max_iter):
            # Update candidate histogram
            patch, mask = get_patch(image, self.x_k, self.box)
            p = extract_histogram(patch, self.bins, self.kernel * mask)
            if self.bg:
                p *= self.bg_weights
            p /= p.sum()

            # Compute backprojection histogram
            p[p < 1e-10] += np.min(p[p > 1e-10]) / 10
            v = np.sqrt((self.bg_weights if self.bg else 1) * self.q / p)

            # Backproject image to obtain mean shift weights
            weights = backproject_histogram(patch, v, self.bins)
            kernel = (self.kernel > 0) * mask
            offset = mean_shift(self.center, weights, kernel) - self.center

            if np.linalg.norm(offset) < self.eps:
                break
            self.x_k += offset

        # Update target model
        q = self.get_target_model(image)
        self.q = (1 - self.alpha) * self.q + self.alpha * q

        left, top = self.x_k - self.box // 2
        return [left, top, *self.box]
