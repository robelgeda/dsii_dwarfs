import warnings

import numpy as np

from scipy import integrate

from astropy.modeling import models
from astropy import modeling



class DwarfSersic2D(modeling.models.Sersic2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalization = self.compute_normalization()

        self.model_1d = models.Sersic1D(self.amplitude, self.r_eff, self.n)

    def compute_normalization(self):
        x = self.x_0.value
        y = self.y_0.value
        delta = self.r_eff.value * 8
        return integrate.dblquad(self, x - delta, x + delta, y - delta, y + delta)[0]

    def compute_npix(self, noise_level, total_flux, max_pix=500):
        x = np.arange(0, max_pix, 1)

        values = total_flux * (self.model_1d(x) / self.normalization)

        if noise_level > values[0]:
            # raise Exception("Noise too large, noise_level = {},  max_galaxy_flux = {}".format(noise_level, values[0]))
            print("Very faint dwarf")
            print("noise_level = {},  max_galaxy_flux = {}".format(noise_level, values[0]))
            return -1

        if noise_level < values[-1]:
            # raise Exception("Noise too small, noise_level = {}, min_galaxy_flux = {}".format(noise_level, values[-1]))
            print("Very bright dwarf")
            print("noise_level = {}, min_galaxy_flux = {}".format(noise_level, values[-1]))
            return max_pix

        argmin = abs(values - noise_level).argmin()

        return int(x[argmin])