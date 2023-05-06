import numpy as np
import quat
import consts


def acc_read(q: np.ndarray) -> np.ndarray:
    """
    Returns the acceleration vector in the body frame

    :param q: quaternion representing R from w to b
    """
    return quat.q_to_rot(q) @ np.array([[0, 0, consts.g]]).T  # type: ignore # m/s^2


def sim_acc_read(q: np.ndarray) -> np.ndarray:
    return acc_read(q) + np.random.randn(3, 1) * consts.sig_a
 