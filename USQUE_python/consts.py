from numpy import pi

# Type of IMU
imu_grade = "tac"  # "tac" or "mem"

a = 1
lam = 1
f = 2 * (a + 1)

N = 12000  # Number of measurements, 2 minutes

# Dimension of state vector, 3 for Rodrigues vector, 3 for bias
n = 6
g = 9.81  # m/s^2

if imu_grade == "tac":
    # Noise params for sensors
    sig_gy_b = 2e-4
    sig_gy_w = 0.16 / 60 / 180 * pi  # 0.16 deg / sqrt(hr)
    sig_acc = 1e-4  # Accel white noise cov m / s / s^(1/2)

elif imu_grade == "mem":
    sig_gy_b = 2e-3
    sig_gy_w = 0.7 / 60 / 180 * pi  # 0.7 deg / sqrt(hr)
    sig_acc = 1e-4  # Accel white noise cov m / s / s^(1/2)

else:
    raise ValueError(f"imu_grade = {imu_grade} is not valid")

# Sampling frequency of the IMU
fs = 20.0  # Hz
dt = 1 / fs  # s
