#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

"""
This script is an implementation of "Unscented Filtering for Spacecraft Attitude Estimation" by Markley and Crassidis

The goal is to estimate the attitude of a spacecraft using a nonlinear filter. The attitude is represented by a quaternion.
The filter uses a 3-vec error quaternion to propagate the state and covariance.
"""

# Initialize everything

