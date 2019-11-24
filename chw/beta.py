"""
Plot the beta distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def Beta(x, alpha, beta):
    pmf = x ** (alpha - 1)  *  (1 - x) ** (beta - 1)
    norm_const = np.trapz(pmf, x)
    pmf /= norm_const
    return pmf

def Gaussian(x, mu, sigma):
    pmf = (1 / np.sqrt(2 * math.pi * sigma ** 2)) * np.exp(- (x - mu ) ** 2 / (2 * sigma ** 2))
    return pmf

# The Beta distribution is defined for a range of x between 0 and 1.
# The smaller the step size in x, the better the normalisation.

x = np.arange(0.0001, 1, 0.0001)    # Exclude 0 and 1 from the range of x for alpha, beta < 1, otherwise we get divide by zero errors

#plt.plot(x, Beta(x, 0.5, 1))        # pmf tends to infinity at x = 0, and 0.5 at x = 1
#plt.plot(x, Beta(x, 0.5, 0.5))      # U-shaped pmf, tending to infinity at x = 0 and x = 1

x = np.arange(0, 1.0001, 0.0001)    # Include 0 and 1 in the range of x for values of alpha and beta >= 1 

#plt.plot(x, Beta(x, 1, 1))          # Uniform pmf, equal to 1 for all x
#plt.plot(x, Beta(x, 2, 1))          # Straight line with positive gradient, from 0 at x = 0 to 2 at x = 1
#plt.plot(x, Beta(x, 1, 2))          # Straight line with negative gradient, from 2 at x = 0 to 0 at x = 1
#plt.plot(x, Beta(x, 2, 2))          # Upside-down parabola centred on x = 0.5
#plt.plot(x, Beta(x, 3, 2))          # Broad, skewed distribution with bump on right
#plt.plot(x, Beta(x, 3, 3))          # Broad symmetric distribution centred on x = 0.5
plt.plot(x, Beta(x, 10, 10))        # Symmetric distribution centred on x = 0.5 (medium width)
#plt.plot(x, Beta(x, 100, 100))      # Narrow symmetric distribution centred on x = 0.5
#plt.plot(x, Beta(x, 30, 100))       # Almost symmetric distribution centred on x ~ 0.226
#plt.plot(x, Beta(x, 30, 50))        # Almost symmetric distribution centred on x ~ 0.372

plt.plot(x, Gaussian(x, 0.5, 0.112)) # Gaussian distribution most similar to Beta(10, 10)

print(np.trapz(Gaussian(x, 0.5, 0.3), x))

plt.show()