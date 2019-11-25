import numpy as np

class DistancePicker:
    """
    Select a random distance for population with a uniform density
    """
    def __init__(self, dmax, dmin=0.):
        """ 
        Initialise: Set the minimum and maximum distance in MPc
        """
        self.dmin = dmin
        self.dmin2 = dmin**2

        self.dmax = dmax
        self.dmax2 = dmax ** 2

    def pick_distance(self):
        r2 = None

        while r2 is None or r2 > self.dmax2 or r2 < self.dmin:
            x = np.random.uniform(self.dmin, self.dmax)
            y = np.random.uniform(self.dmin, self.dmax)
            z = np.random.uniform(self.dmin, self.dmax)
            r2 = x ** 2 + y ** 2 + z ** 2

        r = np.sqrt(r2)
        return r
