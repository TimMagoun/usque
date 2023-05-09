"""
Generate synthetic data for UKF
"""

from typing import Tuple
from quat import prop_matrix
from sensors import acc_read
import numpy as np
from consts import N, dt, sig_gy_w, sig_gy_b, sig_acc
import scipy
from scipy.integrate import cumulative_trapezoid


def gen_data() -> Tuple:
    """
    Returns noisy gyro and accel, as well as GT attitude
    """

    # Generate gt angular velocity
    gt_bias_drift = np.random.randn(N, 3, 1) * sig_gy_b
    gt_bias = cumulative_trapezoid(gt_bias_drift, dx=dt, initial=0, axis=0)
    gt_omega = np.zeros((N, 3, 1))
    num_acc_steps = 5
    for i in range(3):
        acc_fn = scipy.interpolate.interp1d(
            np.arange(num_acc_steps + 1),
            np.random.randn(num_acc_steps + 1) * 1e-1,
            kind="zero",
        )
        gt_omega[:, i, 0] = cumulative_trapezoid(
            acc_fn(np.linspace(0, num_acc_steps, N)), dx=dt, initial=0, axis=0
        )

    assert gt_omega.shape == gt_bias.shape
    gt_omega = gt_bias + gt_omega

    # Generate gt attitude
    gt_q = np.zeros((N, 4, 1))
    gt_q[0, :, 0] = np.array([0, 0, 0, 1])
    for i in range(1, N):
        gt_q[i] = prop_matrix(gt_omega[i]) @ gt_q[i - 1]
        # normalize
        gt_q[i] /= np.linalg.norm(gt_q[i])

    noisy_omega = gt_omega + np.random.randn(N, 3, 1) * sig_gy_w
    noisy_acc = np.zeros((N, 3, 1))
    for i in range(N):
        noisy_acc[i] = acc_read(gt_q[i]) + np.random.randn(3, 1) * sig_acc

    return gt_q, gt_omega, gt_bias, noisy_omega, noisy_acc
