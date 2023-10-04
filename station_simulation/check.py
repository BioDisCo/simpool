from scipy.stats import lognorm, gamma
import numpy as np
import pylab as pl

# mean = 5.1
# stddev = 0.5
# dist=lognorm([stddev], loc=mean)

shape = 5.807
scale = 0.948
dist = gamma(a=shape, scale=scale)

x = np.linspace(0, 20, 1000)
pl.plot(x, dist.pdf(x))
pl.plot(x, dist.cdf(x))


pl.show()
