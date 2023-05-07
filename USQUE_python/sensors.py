import numpy as np
import quat
import consts


def acc_read(q: np.ndarray) -> np.ndarray:
    """
    Returns the acceleration vector in the body frame

    :param q: quaternion representing R from w to b
    """
    return quat.q_to_rot(q) @ np.array([[0, 0, consts.g]]).T  # type: ignore # m/s^2


def Qbar() -> np.ndarray:
    """
    Returns the Qbar matrix as described in Eq. 42

    This matrix is half of the actual discrete process noise covariance described in Eq. 41

    It is added at the beginning of the propagation via Eq. 5a (while generating sigma points)
    and at the end of the propagation via Eq. 8 (during the usual covariance propagation step)
    """

    dt, sig_gy_w, sig_gy_b = consts.dt, consts.sig_gy_w, consts.sig_gy_b

    # Attitude portion of Qbar
    Qbar_att = (sig_gy_w**2 - 1.0 / 6 * sig_gy_b**2 * dt**2) * np.eye(3)
    Qbar_bias = sig_gy_b**2 * np.eye(3)

    Qbar = (
        dt / 2 * np.block([[Qbar_att, np.zeros((3, 3))], [np.zeros((3, 3)), Qbar_bias]])
    )
    return Qbar
