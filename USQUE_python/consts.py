from numpy import pi

# Type of IMU
imu_grade = "mem"  # or "tac" or "nav"

a = 1
lam = 1
f = 2 * (a + 1)

N = 10000  # Number of measurements

# Dimension of state vector, 3 for Rodrigues vector, 3 for bias
n = 6
g = 9.81  # m/s^2

if imu_grade == "mem":
    # Noise params for sensors
    sig_gy_b = 30 / 60 / 180 * pi  # 30 deg/s^1/2
    sig_gy_w = 0.2 / 60 / 180 * pi  # 0.2 deg / sqrt(hr)
    sig_acc = 1e-4  # Accel white noise cov m / s / s^(1/2) aka acceleration random walk

elif imu_grade == "tac":
    # Noise params for sensors
    sig_gy_b = 1.5 / 60 / 180 * pi  # 1.5 deg/s^1/2
    sig_gy_w = 0.16 / 60 / 180 * pi  # 0.16 deg / sqrt(hr)
    sig_acc = 1e-4  # Accel white noise cov m / s / s^(1/2) aka acceleration random walk

elif imu_grade == "nav":
    # Noise params for sensors
    sig_gy_b = 6e-3 / 60 / 180 * pi  # 6e-3 deg/s^1/2
    sig_gy_w = 3e-3 / 60 / 180 * pi  # 3e-3 deg / sqrt(hr)
    sig_acc = 1e-4  # Accel white noise cov m / s / s^(1/2) aka acceleration random walk

else:
    raise ValueError(f"imu_grade = {imu_grade} is not valid")

# Sampling frequency of the IMU
fs = 100.0  # Hz
dt = 1 / fs  # s
