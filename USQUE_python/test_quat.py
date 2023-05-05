import numpy as np
import scipy.spatial.transform as scipy_rot
from quat import Quat


def test_as_rot():
    q = np.array([0.0936586, 0.1873172, 0.2809757, 0.9365858])  # xyzw
    rot_dut = Quat(q).as_rot()
    rot_gt = scipy_rot.Rotation.from_quat(q).as_matrix()

    assert np.allclose(rot_dut, rot_gt)


def test_multiplication():
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    Q1 = Quat(q1)
    Q2 = Quat(q2)

    R12 = (Q1 * Q2).as_rot()
    R12_gt = Q1.as_rot() @ Q2.as_rot()
    print(R12)
    print(R12_gt)
    assert np.allclose(R12, R12_gt)


def test_noncomm():
    q1 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(4)
    q2 /= np.linalg.norm(q2)

    Q1 = Quat(q1)
    Q2 = Quat(q2)

    assert not np.allclose((Q1 * Q2).as_rot(), (Q2 * Q1).as_rot())
