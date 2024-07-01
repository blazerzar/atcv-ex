import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, exp, integrate, symbols


def rw_model(q: float, r: float):
    """Returns Kalman filter matrices for the random walk model. The model is
    defined by the differential equation

        X' = FX + Lw = 0X + Lw = Lw,

    and X' = w. The observation is computed as Y = HX + v and the state is
    defined as [x, y].
    """
    F = Matrix([[0, 0], [0, 0]])
    L = Matrix([[1, 0], [0, 1]])
    A, Q = solve_differential(F, L, q)

    # Measurement takes x, y from the state
    C = np.eye(2, dtype=np.float32)
    R = r * np.eye(2, dtype=np.float32)

    return A, C, Q, R


def ncv_model(q: float, r: float):
    """Returns Kalman filter matrices for the near constant velocity model.
    The model is defined by the differential equation

        X' = FX + Lw,

    and X'' = w. The observation is computed as Y = HX + v and the state is
    defined as [x, y, x', y'].
    """
    F = Matrix(np.vstack([np.eye(4)[2:], np.zeros((2, 4))]))
    L = Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    A, Q = solve_differential(F, L, q)

    # Measurement takes x, y from the state
    C = np.eye(4, dtype=np.float32)[:2]
    R = r * np.eye(2, dtype=np.float32)

    return A, C, Q, R


def nca_model(q: float, r: float):
    """Returns Kalman filter matrices for the near constant velocity model.
    The model is defined by the differential equation

        X' = FX + Lw,

    and X''' = w. The observation is computed as Y = HX + v and the state is
    defined as [x, y, x', y', x'', y''].
    """
    F = Matrix(np.vstack([np.eye(6)[2:], np.zeros((2, 6))]))
    L = Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
    A, Q = solve_differential(F, L, q)

    # Measurement takes x, y from the state
    C = np.eye(6, dtype=np.float32)[:2]
    R = r * np.eye(2, dtype=np.float32)

    return A, C, Q, R


def solve_differential(F: Matrix, L: Matrix, q: float) -> tuple[NDArray, NDArray]:
    """Solves the differential equation X' = FX + Lw."""
    delta = symbols('delta')
    A = exp(F * delta)
    Q = np.array(integrate((A * L) * q * (A * L).T, (delta, 0, 1)), dtype=np.float32)

    return np.array(A.subs(delta, 1), dtype=np.float32), Q
