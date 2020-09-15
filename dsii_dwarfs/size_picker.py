from scipy import stats

class SizePicker:
    """
    There is an empirical size--stellar-mass relation in Misgeld & Hilker 2011 MNRAS 414, 3699 
    (doi:10.1111/j.1365-2966.2011.18669.x), and they provide a table of their input data. Fit a
    linear relation of logRe vs logM* to that and measured the scatter. We use a t distribution
    with 5 degrees of freedom for the scatter to give it more outliers than a Gaussian.
    """
    def __init__(self,noscatter=False):
        self.c0 = 1.74038461
        self.c1 = 0.13943115
        self.sigma = 0.17
        self.noscatter = noscatter
        self.scatter = stats.t(loc=0,scale=self.sigma,df=5)

    def size(self, log_mass):
        """ Given a mass, return a random draw from the size-mass relation """
        if self.noscatter:
            rekpc = self.c1*log_mass+self.c0
        else:
            rekpc = self.c1*log_mass+self.c0 + self.scatter.rvs(len(log_mass))
        return rekpc
