from typing import Union
import numpy as np
import scipy.spatial.transform as scipy_rot


class Quat:
    """
    Simple implementation of a quaternion class, the internal representation is a numpy array of shape (4,).

        q = [x, y, z, w]
    """

    def __init__(self, q: Union[np.ndarray, None] = None) -> None:
        if q is not None:
            self.q = q
        else:
            self.q = np.array([0, 0, 0, 1], dtype=np.float_)

        assert self.q.shape == (4,)
        assert np.isclose(np.linalg.norm(self.q), 1)

    def __repr__(self) -> str:
        return f"Quat({self.q})"

    def __str__(self) -> str:
        return f"{self.q}"

    def __mul__(self, rhs: "Quat") -> "Quat":
        x0, y0, z0, w0 = self.q.tolist()
        x1, y1, z1, w1 = rhs.q.tolist()
        return Quat(
            np.array(
                [
                    x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                    -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                    x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                    -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                ]
            )
        )

    def as_rot(self) -> np.ndarray:
        """
        Returns a 3x3 rotation matrix repr of the quaternion

        Eq. 15
        """
        return scipy_rot.Rotation.from_quat(self.q).as_matrix()

    def pho(self) -> np.ndarray:
        return self.q[0:3]

    def q4(self) -> np.ndarray:
        return self.q[3:]

    def inv(self) -> "Quat":
        return Quat(np.concatenate([-self.pho(), self.q4()]))

    def dq(self, omega: np.ndarray) -> np.ndarray:  # 4x1
        """
        Find derivative of q given omega [wx, wy, xz] in body frame in rad/s
        """
        top_block = self.q4() + skew_sym(self.pho())
        bot_block = -self.pho()
        xi = np.block([[top_block], [bot_block]])  # Eq. 16a
        assert xi.shape == (4, 3)
        return 0.5 * xi @ omega


@staticmethod
def skew_sym(a: np.ndarray) -> np.ndarray:
    return np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float_
    )
