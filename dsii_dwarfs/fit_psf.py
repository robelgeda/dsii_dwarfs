import warnings

import numpy as np

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.table import Table

from matplotlib import pyplot as plt


class FitPSF:
    """
    Class to  fit HSC PSFs
    """
    def __init__(self, filename):
        self.filename = filename

        # Read PSF file
        with fits.open(self.filename) as psf_hdu:
            psf_data = psf_hdu[0].data

        # Generate grid
        self.x, self.y = np.mgrid[:psf_data.shape[0], :psf_data.shape[1]]
        self.z = psf_data

        # Results
        self.model = None
        self.params = None
        self.gamma = None
        self.alpha = None
        self.fwhm = None

    def dofit(self, verbose=True):
        # Fit the data using astropy.modeling
        center_x, center_y = find_center(self.z)

        p_init = models.Moffat2D(x_0=center_x, y_0=center_y)

        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            self.model = fit_p(p_init, self.x, self.y, self.z)

        if verbose:
            print("Mean Residual:", (self.z - self.model(self.x, self.y)).mean())

        self.params = Table(data=[self.model.param_names, self.model.parameters], names=["param_names", "param_vals"])
        self.gamma = self.model.parameters[3]
        self.alpha = self.model.parameters[4]
        self.fwhm = self.model.fwhm

    def plot_results(self):
        # Plot the data with the best-fit model
        vmin, vmax = np.percentile(self.z, 1.), np.percentile(self.z, 99.)
        print(vmin, vmax)
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.z, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(self.model(self.x, self.y), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(100. * (self.z - self.model(self.x, self.y)) / self.z, origin='lower', interpolation='nearest',
                   vmin=0, vmax=100.)
        plt.title("Residual in percent")


def find_center(array):
    return (np.array(array.shape) - 1) / 2
