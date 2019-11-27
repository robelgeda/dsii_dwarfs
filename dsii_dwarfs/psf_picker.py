import warnings
import numpy as np

from astropy.io import fits
from astropy.modeling import models
from scipy import stats

class HSCMoffatPSFPicker:
    """
    Create a picker for the PSF. The fwhm routine is just for validating
    that the distribution looks okay in comparison to the Subaru data. The
    get_oversampled_psf routine returns the oversampled PSF. The native
    Subaru pixel scale is 0.17 arcsec/pixels and these are oversampled by default
    by a factor of 4. We're feeding it a FITS file with this oversampling as
    a template so we can get a WCS on it, for what it is worth. We will not
    bother writing out the file, but just return the Moffat parameters gamma
    (in arcsec) and alpha so we can put them in a table.
    
    The parameters of the gumbel distribution that match the Subaru seeing
    FWHM distribution were determined by trial and error.
    """

    def __init__(self,
                 oversampled_size=125,
                 oversampling=5,
                 gamma0=2.3,
                 alpha0=1.9,
                 sigma_log_gamma=0.15,
                 max_gamma_factor=4.):

        self.x0,self.y0 = oversampled_size/2,oversampled_size/2

        self.x,self.y = np.mgrid[:oversampled_size, :oversampled_size]

        self.oversampling = oversampling

        self.c1 = 0.30

        self.sigma = sigma_log_gamma
        self.gamma0 = gamma0*oversampling
        self.alpha = alpha0
        self.max_gamma = max_gamma_factor*oversampling

        # With a bit more tail than Gaussian; truncate at some max
        self.gscatter = stats.gumbel_r(loc=np.log10(self.gamma0),scale=self.sigma) 

    def find_center(self, array):
        return (np.array(array.shape)-1)/2

    def gamma(self, size=1):
        """ Gumbell distribution truncated at a max value """

        gammas = []
        for i in range(size):
            g = self.max_gamma
            while g >= self.max_gamma:
                g = 10.**self.gscatter.rvs(size=1)[0]
            gammas += [g]
        return np.array(gammas)

    def fwhm(self,size=1):
        gammas = self.gamma(size=size)
        fwhm = []
        for g in gammas:
            m2d = models.Moffat2D(x_0=self.x0,y_0=self.y0,gamma=g,alpha=self.alpha)
            fwhm += [m2d.fwhm/self.oversampling]
        return np.array(fwhm)

    def get_oversampled_psf(self, verbose=True):
        gamma = self.gamma(size=1)[0]
        alpha = self.alpha
        if verbose:
            print(self.x0,self.y0,gamma,alpha)
        m2d = models.Moffat2D(x_0=self.x0,y_0=self.y0,gamma=gamma,alpha=alpha)
        psf = m2d(self.x,self.y) 
        return gamma,alpha,psf
