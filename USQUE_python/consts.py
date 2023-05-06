# Parameters for fine-tuning filter. Crassidis and Markley recommmend
#   a=1,lambda=1
# Try (a,lambda) values (1,1), (1,0), (1,-1), and (2,-1)
a = 1
lam = 1
f = 2 * (a + 1)

# Dimension of state vector, 3 for Rodrigues vector, 3 for bias
n = 6
g = 9.81  # m/s^2
# Noise params for sensors
# ! TODO: Find actual values
sig_b = 1e-6  # Gyro bias cov rad / s^(3/2)
sig_w = 1e-6  # Gyro white noise cov rad / s^(1/2)
sig_a = 1e-6  # Accel white noise cov m / s^(5/2)
