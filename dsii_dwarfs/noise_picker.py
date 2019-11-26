import numpy as np
from scipy import stats

class NoisePicker:
    """ Set up the parameters of a Gaussian noise model. 
        Return deviates from it.
    """
    def __init__(self, sigma=0.02, sigma_range=None):
        if sigma_range:
            self.min,self.max = sigma_range
        else:
            self.min,self.max = sigma,sigma

        self.sigma = None
        self.scatter = None

    def pick_noise(self, size):
        """ Return an array of random deviates of a given size"""
        self.sigma = np.random.uniform(self.min,self.max)
        self.scatter = stats.norm(loc=0,scale=self.sigma)
        return self.scatter.rvs(size=size)
