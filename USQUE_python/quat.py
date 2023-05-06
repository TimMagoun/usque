import numpy as np
import scipy.spatial.transform as scipy_rot
import consts


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


def rho(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[0:3]


def q4(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[3]


def q_inv(q: np.ndarray) -> np.ndarray:
    check_q(q)
    q_inv = q.copy()
    q_inv[0:3] *= -1
    return q_inv


def qdot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:  # 4x1
    """
    Find derivative of q given omega [wx, wy, xz] in body frame in rad/s

    Eq. 19
    """
    check_q(q)
    assert omega.shape == (3,)
    top_block = q4(q) + skew_sym(rho(q))
    bot_block = -rho(q)
    xi = np.block([[top_block], [bot_block]])  # Eq. 16a
    assert xi.shape == (4, 3)
    return 0.5 * xi @ omega


def q_to_rod(q: np.ndarray) -> np.ndarray:
    """
    Returns the Rodrigues vector of the given quaternion
    """
    check_q(q)
    # Eq 20
    return consts.f * rho(q) / (consts.a + q4(q))


def rod_to_q(rod: np.ndarray) -> np.ndarray:
    """
    Returns the quaternion of the given Rodrigues vector
    """
    assert rod.shape == (3,)
    # Eq 21
    rod_norm_sq = np.linalg.norm(rod) ** 2
    q4_r = (
        -consts.a * rod_norm_sq
        + consts.f * np.sqrt(consts.f**2 + (1 - consts.a) ** 2 * rod_norm_sq)
    ) / (consts.f**2 + rod_norm_sq)

    pho_r = (consts.a + q4_r) * rod / consts.f

    return np.array([pho_r[0], pho_r[1], pho_r[2], q4_r])


def skew_sym(a: np.ndarray) -> np.ndarray:
    return np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float_
    )
