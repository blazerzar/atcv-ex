from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ex2_utils import extract_histogram, get_patch
from ex3_utils import create_cosine_window
from kalman_filter import nca_model, ncv_model, rw_model

MOTION_MODELS = ('rw', 'ncv', 'nca')


@dataclass
class PFParams:
    motion_model: str = 'ncv'
    num_particles: int = 100
    q: float = 1.0
    sigma: float = 0.05
    alpha: float = 0.05
    num_bins: int = 16


class ParticleFilterTracker:
    def __init__(self, params: PFParams):
        """Create a particle filter tracker.

        Parameters:
            - motion_model: Motion model to use (rw, ncv, nca)
            - num_particles: Number of particles to use
            - q: Dynamic model matrix Q parameter
            - sigma: Standard deviation of observation model Gaussian
            - alpha: Update rate of the visual model
            - num_bins: Number of bins for the histogram
        """
        if params.motion_model not in MOTION_MODELS:
            raise ValueError('Invalid motion model')
        self.motion_model = params.motion_model
        self.Phi, _, self._Q, _ = [rw_model, ncv_model, nca_model][
            MOTION_MODELS.index(params.motion_model)
        ](1, 1)

        self.num_particles = params.num_particles
        self.q = params.q
        self.sigma = params.sigma
        self.alpha = params.sigma
        self.num_bins = params.num_bins
        self.rand = np.random.default_rng(seed=1)

    def initialize(
        self, image: NDArray, region: tuple[float, float, float, float]
    ) -> None:
        """Initialize the tracker with the first frame and the region to track.

        Parameters:
            - image: Image to initialize the tracker
            - region: Initial region to track (left, top, width, height)
        """
        left, top, width, height = region
        center_y, center_x = top + height / 2, left + width / 2
        self.x_k = np.array([center_x, center_y])
        self.box = np.array([width, height])
        self.Q = self._Q * self.q * np.min(self.box)

        self.particles = self.sample_particles(center_x, center_y)

        # Initialize visual model
        patch, _ = get_patch(image, (center_x, center_y), self.box)
        h = create_cosine_window(patch.shape[-2::-1])
        self.model = extract_histogram(patch, self.num_bins, weights=h)
        self.model = self.model / np.sum(self.model)

    def track(self, image: NDArray) -> tuple[float, float, float, float]:
        """Track the region in the next frame."""
        # Move by the dynamic model
        self.particles = self.Phi @ self.particles
        self.particles += self.rand.multivariate_normal(
            np.zeros(len(self.Q)), self.Q, self.num_particles
        ).T

        histograms = np.zeros((self.num_particles, self.model.size))
        for i, particle in enumerate(self.particles.T):
            x, y = particle[:2]
            # Out-of-bounds particle
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                histograms[i] = np.zeros(self.model.size)
                continue
            patch, mask = get_patch(image, particle[:2], self.box)
            histograms[i] = extract_histogram(patch, self.num_bins, weights=mask)

        # Remove invalid particles
        valid = np.sum(histograms, axis=1) > 0
        histograms = histograms[valid]
        self.particles = self.particles[:, valid]

        if np.sum(valid) == 0:
            self.particles = self.sample_particles(self.x_k[0], self.x_k[1])
            return *(self.x_k - self.box / 2), *self.box  # type: ignore

        # Compute weight for each particle
        histograms /= np.sum(histograms, axis=1, keepdims=True)
        hellinger = 2 - 2 * np.sqrt(histograms) @ np.sqrt(self.model)
        weights = np.exp(-hellinger / (2 * self.sigma**2))
        weights /= np.sum(weights)

        self.x_k = np.sum(self.particles * weights, axis=1)[:2]
        left, top = self.x_k - self.box / 2

        # Resample particles according to their weights
        samples = self.rand.choice(
            np.arange(self.particles.shape[1]),
            self.num_particles,
            p=weights,
        )
        self.particles = self.particles[:, samples]

        # Update visual model
        patch, mask = get_patch(image, self.x_k, self.box)
        h = create_cosine_window(patch.shape[-2::-1])
        model = extract_histogram(patch, self.num_bins, weights=mask * h)
        model = model / np.sum(model)
        self.model = (1 - self.alpha) * self.model + self.alpha * model

        return left, top, *self.box  # type: ignore

    def sample_particles(self, x: float, y: float):
        particles = np.vstack(
            [
                self.rand.normal(x, self.box[0] / 6, self.num_particles),
                self.rand.normal(y, self.box[1] / 6, self.num_particles),
            ]
        )

        # For NCV and NCA, the state also includes velocities and accelerations
        if self.motion_model.startswith('nc'):
            rows = [2, 4][self.motion_model == 'nca']
            vel_acc = np.zeros((rows, self.num_particles))
            particles = np.vstack([particles, vel_acc])

        return particles
