import numpy as np
import scipy.spatial.transform as scipy_rot
import consts

'''
Q = XYZW!!
'''


def norm_q(q: np.ndarray) -> np.ndarray:
    """
    Normalize the given quaternion
    """
    return q / np.linalg.norm(q)


def check_q(q: np.ndarray) -> None:
    assert q.shape == (4, 1), f"q.shape = {q.shape}"
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-4), f"q = {q}, norm = {np.linalg.norm(q)}"


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    check_q(q1)
    check_q(q2)

    x0, y0, z0, w0 = q1[:, 0].tolist()
    x1, y1, z1, w1 = q2[:, 0].tolist()
    q_res = np.array(
        [
            [
                x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
            ]
        ]
    ).T

    return norm_q(q_res)


def q_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix repr of the given

    Eq. 15
    """
    check_q(q)
    return scipy_rot.Rotation.from_quat(q[:, 0]).as_matrix()


def rho(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[0:3, :]


def q4(q: np.ndarray) -> np.ndarray:
    check_q(q)
    return q[3, :]


def q_inv(q: np.ndarray) -> np.ndarray:
    check_q(q)
    q_inv = q.copy()
    q_inv[0:3, :] *= -1

    return norm_q(q_inv)


def q_to_rod(q: np.ndarray) -> np.ndarray:
    """
    Returns the Rodrigues vector (3, 1) of the given quaternion
    """
    check_q(q)
    # Eq 20
    return consts.f * rho(q) / (consts.a + q4(q))  # Ensure returning a (3,1)


def rod_to_q(rod: np.ndarray) -> np.ndarray:
    """
    Returns the quaternion of the given Rodrigues vector
    """
    assert rod.shape == (3, 1), f"rod.shape = {rod.shape}"
    # Eq 21
    rod_norm_sq = np.linalg.norm(rod) ** 2
    q4_r = (
        -consts.a * rod_norm_sq
        + consts.f * np.sqrt(consts.f**2 + (1 - consts.a) ** 2 * rod_norm_sq)
    ) / (consts.f**2 + rod_norm_sq)

    pho_r = (consts.a + q4_r) * rod / consts.f
    q_res = np.array([[pho_r[0, 0], pho_r[1, 0], pho_r[2, 0], q4_r]]).T
    return norm_q(q_res)


def prop_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Propagation model for a given omega

    Eq. 28
    """
    assert omega.shape == (3, 1)
    omega_norm = np.linalg.norm(omega)
    if omega_norm == 0:
        return np.eye(4)

    psi_k = psi(omega)
    Omega_A = np.cos(0.5 * omega_norm * consts.dt) * np.eye(3) - skew_sym(psi_k)
    Omega_B = psi_k
    Omega_C = -psi_k.T
    Omega_D = np.cos(0.5 * omega_norm * consts.dt)

    return np.block([[Omega_A, Omega_B], [Omega_C, Omega_D]])


def psi(omega: np.ndarray) -> np.ndarray:
    """
    Eq. 29
    """
    assert omega.shape == (3, 1)
    omega_norm = np.linalg.norm(omega)
    assert omega_norm != 0
    psi = np.sin((0.5 * omega_norm * consts.dt) * omega / omega_norm)
    return psi  # 3x1


def skew_sym(a: np.ndarray) -> np.ndarray:
    return np.array(
        [[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]],
        dtype=np.float_,
    )
