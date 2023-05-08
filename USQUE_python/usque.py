#! /usr/bin/env python3
import numpy as np
from quat import norm_q, q_to_rod, q_mul, q_inv, rod_to_q, prop_matrix, check_q
import sensors
from consts import n, lam, sig_acc, N
from tqdm import tqdm

np.set_printoptions(precision=4)
DEFAULT_TYPE = np.float64

"""
This script is an implementation of "Unscented Filtering for Spacecraft Attitude Estimation" by Markley and Crassidis

The goal is to estimate the attitude of a spacecraft using a nonlinear filter.
The attitude is represented by a quaternion. The filter uses a 3-vec error
quaternion to propagate the state and covariance.
"""


def run_ukf(x0, P0, W, Y):
    def propagate(k: int, w_k: np.ndarray):
        # 1. Calculate sigma points
        Chi_k = np.zeros((2 * n + 1, n, 1))  # 13 x 6 sigma points
        Pk = P_p[k]

        mat = (n + lam) * (Pk + Qbar)
        # w, v = np.linalg.eig(Pk)
        # print(w)

        sig_k = np.linalg.cholesky(mat)  # Eq. 5a
        assert np.allclose(sig_k @ sig_k.T, mat)

        Chi_k[0] = X_p[k]
        assert np.allclose(Chi_k[0, :3], np.zeros((3, 1)))
        for j in range(n):
            Chi_k[1 + j] = X_p[k] + sig_k[:, j : j + 1]  # sig_k column must be (6, 1)
            Chi_k[1 + j + n] = X_p[k] - sig_k[:, j : j + 1]
        # 2. Calculate quaternions of sigma points (Eq. 32 a, b)
        # Prepares for propagation with angular velocity, q_sample = q_hat_k_plus
        q_sample = np.zeros((2 * n + 1, 4, 1))
        for j in range(2 * n + 1):
            q_sample[j] = q_mul(rod_to_q(Chi_k[j, :3]), q_p[k])  # Eq. 32b
            check_q(q_sample[j])
        # 3. Propagate sampled quaternions
        # q_prop = q_hat_k+1_minus
        omega_k = np.zeros((2 * n + 1, 3, 1))  # Sampled omegas based on sampled biases
        assert w_k.shape == Chi_k[0, 3:].shape, f"{w_k.shape} {Chi_k[0, 3:].shape}"

        for j in range(2 * n + 1):
            omega_k[j] = w_k - Chi_k[j, 3:]  # Eq. 35
            # print(omega_k[j])

        for j in range(2 * n + 1):
            # print(omega_k[j])
            q_kp1_m[k + 1, j] = prop_matrix(omega_k[j]) @ q_sample[j]  # Eq. 34
            q_kp1_m[k + 1, j] = norm_q(q_kp1_m[k + 1, j])
            # print(q_sample_kp1[j])

        # 4. Convert propagated quaternions back into error quaternions in rodregues form
        # The biases stay the same
        Chi_kp1_m[k + 1, :, 3:] = Chi_k[:, 3:]  # Eq. 38
        # Error quaternions in rodregues form, Chi_k+1, note: Chi_k_prop[0, :3] is always 0 (mean has no err)
        q_kp1_mean_inv = q_inv(q_kp1_m[k + 1, 0])
        assert np.allclose(
            q_to_rod(q_mul(q_kp1_mean_inv, q_kp1_m[k + 1, 0])), np.array([[0, 0, 0]]).T
        )
        for j in range(2 * n + 1):
            Chi_kp1_m[k + 1, j, :3] = q_to_rod(
                q_mul(q_kp1_m[k + 1, j], q_kp1_mean_inv)
            )  # Eq. 36, 37a, 37b

        # 5. Calculate mean and covariance of propagated sigma points
        X_m[k + 1] = lam * Chi_kp1_m[k + 1, 0] + 0.5 * np.sum(
            Chi_kp1_m[k + 1, 1:], axis=0, keepdims=True
        )
        X_m[k + 1] *= 1.0 / (n + lam)  # Eq. 7
        # print(X_m[k + 1])
        # Eq. 8
        P_m_0 = (
            lam
            * (Chi_kp1_m[k + 1, 0] - X_m[k + 1])
            @ (Chi_kp1_m[k + 1, 0] - X_m[k + 1]).T
        )
        P_m[k + 1] = P_m_0
        for j in range(1, 2 * n + 1):
            P_m[k + 1] += (
                0.5
                * (Chi_kp1_m[k + 1, j] - X_m[k + 1])
                @ (Chi_kp1_m[k + 1, j] - X_m[k + 1]).T
            )

        P_m[k + 1] *= 1.0 / (n + lam)
        P_m[k + 1] += Qbar  # Eq. 8

    def update(k: int, y_k: np.ndarray):
        gamma = np.zeros((2 * n + 1, 3, 1))  # Eq 10, all simulated measurments
        for j in range(2 * n + 1):
            gamma[j] = sensors.acc_read(q_kp1_m[k + 1, j])

        y_m = lam * gamma[0]
        y_m += 0.5 * np.sum(gamma[1:], axis=0)
        y_m *= 1.0 / (n + lam)  # Eq. 9
        # y_m should be (3, 1)

        # Output cov (Eq. 11)
        Pyy = lam * (gamma[0] - y_m) @ (gamma[0] - y_m).T
        for j in range(1, 2 * n + 1):
            Pyy += 0.5 * (gamma[j] - y_m) @ (gamma[j] - y_m).T
        Pyy *= 1.0 / (n + lam)
        # Innovation cov
        Pvv = Pyy + np.eye(3) * sig_acc**2  # Eq. 12

        # Cross cov
        Pxy = lam * (Chi_kp1_m[k + 1, 0] - X_m[k + 1]) @ (gamma[0] - y_m).T
        for j in range(1, 2 * n + 1):
            Pxy += 0.5 * (Chi_kp1_m[k + 1, j] - X_m[k + 1]) @ (gamma[j] - y_m).T
        Pxy *= 1.0 / (n + lam)

        # Kalman gain
        K = Pxy @ np.linalg.inv(Pvv)  # Eq. 4
        # Calculate innovation
        v = y_k - y_m  # Eq. 3
        # print(f"y_k = {y_k}, y_m = {y_m}")
        # print(f"v = {v}")
        X_p[k + 1] = X_m[k + 1] + K @ v  # Eq. 2a
        P_p[k + 1] = P_m[k + 1] - K @ Pvv @ K.T  # Eq. 2b
        # Calculate q_p, transfers information from X_p to q_p
        q_p[k + 1] = q_mul(rod_to_q(X_p[k + 1, :3]), q_kp1_m[k + 1, 0])  # Eq. 13
        # Set error quaternion part of X_p to 0
        X_p[k + 1, :3] = 0

    X_p = np.zeros(
        (N, n, 1), dtype=DEFAULT_TYPE
    )  # State estimates X+ [dp, b] after update
    X_m = np.zeros(
        (N, n, 1), dtype=DEFAULT_TYPE
    )  # State estimates X- [dp, b] before update
    Chi_kp1_m = np.zeros(
        (N, 2 * n + 1, n, 1), dtype=DEFAULT_TYPE
    )  # Propagated sigma points
    X_p[0] = x0
    P_p = np.zeros((N, n, n), dtype=DEFAULT_TYPE)  # Covariance estimates after update
    P_m = np.zeros((N, n, n), dtype=DEFAULT_TYPE)  # Covariance estimates before update
    P_p[0] = P0
    q_p = np.zeros((N, 4, 1), dtype=DEFAULT_TYPE)  # Quaternion estimates after update
    q_p[0, 3] = 1  # Initial is identity
    q_kp1_m = np.zeros(
        (N, 2 * n + 1, 4, 1), dtype=DEFAULT_TYPE
    )  # All sampled quaternions after propagation, used in the update step

    Qbar = sensors.Qbar()
    # Loop through measurements
    for k in tqdm(range(N - 1)):  # N-1 measurements, N states
        # Propagate #
        propagate(k, W[k])
        # Update #
        update(k, Y[k])

    # plt.plot(q_p[:, 2], label="dp")
    # plt.show()
    return q_p, P_p


if __name__ == "__main__":
    data = np.load("/home/tim/Documents/usque/USQUE_python/data.npz")
    # Initialize everything
    x0 = np.array([[0, 0, 0, 0, 0, 0]], dtype=DEFAULT_TYPE).T
    # P0 = diag([attitude err cov, bias err cov])
    P0 = np.eye(n, dtype=DEFAULT_TYPE) * 1e-1  # TODO double check

    Y = data["noisy_acc"]  # IMU Accel observations
    W = data["noisy_omega"]  # IMU Gyro observations

    assert Y.shape == (N, 3, 1)
    assert W.shape == (N, 3, 1)

    run_ukf(x0, P0, W, Y)
