import numpy as np
import scipy.spatial.transform as scipy_rot


def check_q(q: np.ndarray) -> None:
    assert q.shape == (4,)
    assert np.isclose(np.linalg.norm(q), 1.0)


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    check_q(q1)
    check_q(q2)

    x0, y0, z0, w0 = q1.tolist()
    x1, y1, z1, w1 = q2.tolist()
    return np.array(
        [
            x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
            -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
            x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
        ]
    )


def q_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix repr of the given

    Eq. 15
    """
    check_q(q)
    return scipy_rot.Rotation.from_quat(q).as_matrix()


def pho(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[0:3]


def q4(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[3:]


def q_inv(q: np.ndarray) -> np.ndarray:
    check_q(q)
    q_inv = q.copy()
    q_inv[0:3] *= -1
    return q_inv


def dq(q: np.ndarray, omega: np.ndarray) -> np.ndarray:  # 4x1
    """
    Find derivative of q given omega [wx, wy, xz] in body frame in rad/s
    """
    check_q(q)
    assert omega.shape == (3,)
    top_block = q4(q) + skew_sym(pho(q))
    bot_block = -pho(q)
    xi = np.block([[top_block], [bot_block]])  # Eq. 16a
    assert xi.shape == (4, 3)
    return 0.5 * xi @ omega


def skew_sym(a: np.ndarray) -> np.ndarray:
    return np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float_
    )
