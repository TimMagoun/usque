# Parameters for fine-tuning filter. Crassidis and Markley recommmend
#   a=1,lambda=1
# Try (a,lambda) values (1,1), (1,0), (1,-1), and (2,-1)
a = 1
lam = 1
f = 2 * (a + 1)

# Dimension of state vector, 3 for Rodrigues vector, 3 for bias
n = 6

# Noise params for sensors
# ! TODO: Find actual values
sig_b = 1e-6  # Gyro bias cov
sig_w = 1e-6  # Gyro white noise cov
sig_a = 1e-6  # Accel white noise cov
