from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ex1_utils import gaussderiv, gausssmooth


def lucaskanade(
    im1: NDArray, im2: NDArray, N: int, harris=False
) -> tuple[NDArray, NDArray]:
    """
    Compute the optical flow between two images using the Lucas-Kanade method.

    Parameters:
        - im1: A grayscale image of shape (H, W)
        - im2: A grayscale image of shape (H, W)
        - N: The size of the neighborhood window
        - harris: Whether to use the Harris corner detector for improvement

    Returns:
        - u: A 2D array of shape (H, W) representing the x-component of the
             optical flow vectors
        - v: A 2D array of shape (H, W) representing the y-component of the
             optical flow vectors
    """
    # Derivatives
    derivatives = np.array([gaussderiv(im, 1.0) for im in (im1, im2)])
    I_x = np.mean(derivatives[:, 0], axis=0)
    I_y = np.mean(derivatives[:, 1], axis=0)
    I_t = gausssmooth(im2 - im1, 1.0)

    kernel = np.ones((N, N))

    # Derivatives for covariance matrix
    I_x2 = cv2.filter2D(I_x**2, -1, kernel)
    I_y2 = cv2.filter2D(I_y**2, -1, kernel)
    I_xy = cv2.filter2D(I_x * I_y, -1, kernel)
    I_xt = cv2.filter2D(I_x * I_t, -1, kernel)
    I_yt = cv2.filter2D(I_y * I_t, -1, kernel)

    # Compute optical flow
    det = I_x2 * I_y2 - I_xy**2
    u = I_xy * I_yt - I_y2 * I_xt
    u = np.divide(u, det, out=np.zeros_like(u), where=det != 0)
    v = I_xy * I_xt - I_x2 * I_yt
    v = np.divide(v, det, out=np.zeros_like(v), where=det != 0)

    if harris:
        k = 0.06
        h = det - k * (I_x2 + I_y2) ** 2
        u[h < 1e-15] = 0
        v[h < 1e-15] = 0

    return u, v


def hornschunck(
    im1: NDArray,
    im2: NDArray,
    n_iters: int = 1000,
    lmbd: float = 0.5,
    tol: Optional[float] = None,
    lk_init: bool = False,
    convergence: Optional[list[int]] = None,
) -> tuple[NDArray, NDArray]:
    """
    Compute the optical flow between two images using the Horn-Schunck method.

    Parameters:
        - im1: A grayscale image of shape (H, W)
        - im2: A grayscale image of shape (H, W)
        - n_iters: Number of iterations
        - lmbd: Horn-Schunck parameter
        - tol: Tolerance for convergence
        - lk_init: Whether to use the Lucas-Kanade method for initialization
        - convergence: A list to store the iteration number at which the
                       algorithm converges

    Returns:
        - u: A 2D array of shape (H, W) representing the x-component of the
             optical flow vectors
        - u: A 2D array of shape (H, W) representing the y-component of the
             optical flow vectors
    """
    # Derivatives
    derivatives = np.array([gaussderiv(im, 1.0) for im in (im1, im2)])
    I_x = np.mean(derivatives[:, 0], axis=0)
    I_y = np.mean(derivatives[:, 1], axis=0)
    I_t = gausssmooth(im2 - im1, 1.0)

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    # Optical flow is initialized to zero by default
    if lk_init:
        u, v = lucaskanade(im1, im2, 7, harris=True)
        u, v = np.clip(u, -2, 2), np.clip(v, -2, 2)
        u, v = gausssmooth(u, 1.0), gausssmooth(v, 1.0)
    else:
        u, v = np.zeros_like(im1), np.zeros_like(im1)

    # Iteratively solve for optical flow
    for i in range(n_iters):
        u_a = cv2.filter2D(u, -1, kernel)
        v_a = cv2.filter2D(v, -1, kernel)
        P = I_x * u_a + I_y * v_a + I_t
        D = lmbd + I_x**2 + I_y**2

        u_ = u_a - I_x * (P / D)
        v_ = v_a - I_y * (P / D)

        # Check for convergence if a tolerance is given
        if tol is not None and np.sum((u - u_) ** 2) + np.sum((v - v_) ** 2) < tol:
            if convergence is not None and len(convergence) > 0:
                convergence[0] = i
            break

        u, v = u_, v_

    return u, v


def lucaskanade_pyramids(
    im1: NDArray, im2: NDArray, N: int, min_size: int = 4, repeat: int = 5
) -> tuple[NDArray, NDArray]:
    """
    Compute the optical flow between two images using the Lucas-Kanade method
    with Gaussian pyramids improvement.

    Parameters:
        - im1: A grayscale image of shape (H, W)
        - im2: A grayscale image of shape (H, W)
        - N: The size of the neighborhood window
        - min_size: The minimum size of the Gaussian pyramid
        - repeat: The number of times to repeat the Lucas-Kanade method

    Returns:
        - u: A 2D array of shape (H, W) representing the x-component of the
             optical flow vectors
        - v: A 2D array of shape (H, W) representing the y-component of the
             optical flow vectors
    """
    # Compute Gaussian pyramids down to the minimum size
    pyramids = [[im1, im2]]
    levels = np.floor(np.log2(min(im1.shape) / min_size)).astype(int)
    for _ in range(levels):
        pyramids.append([cv2.pyrDown(p) for p in pyramids[-1]])

    u, v = np.zeros_like(pyramids[-1][0]), np.zeros_like(pyramids[-1][0])
    for i, (im1, im2_) in enumerate(reversed(pyramids)):
        # Upsample the optical flow from the previous level
        if i > 0:
            # Distances are doubled when upsampling
            u = 2 * cv2.resize(u, im1.shape[::-1])
            v = 2 * cv2.resize(v, im1.shape[::-1])

        for _ in range(repeat):
            im2 = im2_.copy()
            im2 = cv2.remap(
                im2_,
                (-u + np.arange(u.shape[1])).astype(np.float32),
                (-v + np.arange(u.shape[0]).reshape(-1, 1)).astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )

            u_, v_ = lucaskanade(im1, im2, N)
            u, v = u + u_, v + v_

    return u, v
