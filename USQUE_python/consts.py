# Parameters for fine-tuning filter. Crassidis and Markley recommmend
#   a=1,lambda=1
# Try (a,lambda) values (1,1), (1,0), (1,-1), and (2,-1)
a = 1
lam = 1
f = 2 * (a + 1)

N = 10000  # Number of measurements

# Dimension of state vector, 3 for Rodrigues vector, 3 for bias
n = 6
g = 9.81  # m/s^2

# Noise params for sensors
sig_gy_b = 1e-4  # Gyro bias cov rad / s^(3/2)
sig_gy_w = 2e-3  # Gyro white noise cov rad / s^(1/2)
sig_acc = 1e-4  # Accel white noise cov m / s^(5/2)

# Sampling frequency of the IMU
fs = 100.0  # Hz
dt = 1 / fs  # s
