import os
import numpy as np
from scipy import stats
from scipy import integrate
from scipy import interpolate
from astropy.table import Table
from scipy.stats import rv_continuous
from astropy.io import ascii

from . import DATA_PATH

# Create a function that returns the PDF of the axial ratio
# Empirical virgo-cluster dE distribution taken from
# Sanchez-Janssen 2016, ApJ 820, 69 doi:10.3847/0004-637X/820/1/69
# Normalize PDF via numerical integral (it was pretty close to normalized to begin with)
# While it's not necessary to add the _cdf routine, it speeds up the rvs draws by a huge factor
class AxialRatioPDF(rv_continuous):
    """
    The PDF of the axial ratio of dwarf-elliptical galaxies
    This is a subclass of scipy.stats.rv_continuous, so the rvs() method returns the
    random deviates and the pdf() method returns the probability of getting a certain axial ratio
    The empirical virgo-cluster dE distribution taken from
    Sanchez-Janssen 2016, ApJ 820, 69 doi:10.3847/0004-637X/820/1/69
    Normalize PDF via numerical integral (it was pretty close to normalized to begin with)
    While it's not necessary to add the _cdf routine, it speeds up the rvs draws by a huge factor
    """

    def __init__(self, **args):
        super(AxialRatioPDF,self).__init__(a=0., b=1., **args)

        #self.qdist = Table.read('data/sanchez-janssen_fig9.txt',format='ascii.commented_header')
        self.qdist = sanchez_jansen()

        self.normalization = integrate.trapz(self.qdist['pdf'],self.qdist['q'])

        self.qfunc = interpolate.interp1d(self.qdist['q'],self.qdist['pdf'],kind='linear')

        qsamples = np.arange(0,1.01,0.01)
        cdf_samples = np.array([integrate.quad(self._pdf, 0, q,
                                               limit=6000, epsabs=1.e-4,
                                               epsrel=1.e-4)[0] for q in qsamples])
        self.cfunc = interpolate.interp1d(qsamples,cdf_samples)

    def _pdf(self,q):
        return self.qfunc(q)/self.normalization

    def _cdf(self,q):
        return self.cfunc(q)

def sanchez_jansen():
    """ Data from  Sanchez-Janssen 2016, ApJ 820, 69 doi:10.3847/0004-637X/820/1/69 """
    sj = os.path.join(DATA_PATH, 'sanchez-janssen_fig9.txt')
    t = Table.read(sj, format='ascii.commented_header')
    return t


