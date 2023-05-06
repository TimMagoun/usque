import numpy as np
import scipy.spatial.transform as scipy_rot
from quat import q_to_rod, q_to_rot, q_mul, rod_to_q


def test_as_rot():
    q = np.array([0.0936586, 0.1873172, 0.2809757, 0.9365858])  # xyzw
    rot_dut = q_to_rot(q)
    rot_gt = scipy_rot.Rotation.from_quat(q).as_matrix()

    assert np.allclose(rot_dut, rot_gt)


def test_multiplication():
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    R12 = q_to_rot(q_mul(q1, q2))
    R12_gt = q_to_rot(q1) @ q_to_rot(q2)
    print(R12)
    print(R12_gt)
    assert np.allclose(R12, R12_gt)


def test_rod():
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    q2 = rod_to_q(q_to_rod(q))
    assert np.allclose(q, q2), f"{q} != {q2}"
