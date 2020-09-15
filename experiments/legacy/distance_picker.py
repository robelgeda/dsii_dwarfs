import numpy as np

class DistancePicker:
    """
    Select a random distance for population with a uniform density
    """
    def __init__(self,dmax):
        """ 
        Initialise: Set the maximum distance 
        """
        self.dmax = dmax
        self.dmax2 = dmax**2
    def pick_distance(self):
        x = np.random.uniform(0,self.dmax)
        y = np.random.uniform(0,self.dmax)
        z = np.random.uniform(0,self.dmax)
        r2 = x**2 + y**2 + z**2
        r = np.sqrt(r2)
        if r2 > self.dmax2:
            r = self.pick_distance()
        return r
