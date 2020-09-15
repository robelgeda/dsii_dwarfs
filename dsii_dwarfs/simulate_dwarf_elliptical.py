# Create a bunch of fake dE galaxies
import os

import numpy as np

from scipy import integrate, stats

from astropy.cosmology import Planck15 as cosmo

from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.table import Table

from webbpsf import WFI

import imf  # https://github.com/keflavich/imf

from . import DATA_PATH
from .download_hsc_psf import HSC_PSF_DOWNLOADERS
from .axial_ratio_picker import AxialRatioPDF
from .distance_picker import DistancePicker
from .isochrone_picker import IsochronePicker
from .dwarf_sersic_2D import DwarfSersic2D
from .schechter_picker import schechter_picker
from .size_picker import SizePicker
from .fit_psf import FitPSF
from .noise_picker import NoisePicker
from .psf_picker import HSCMoffatPSFPicker


def observed_cts_per_sec(isochrone_flux, lumdist, zpt=27.0):
    zptflux = 10. ** (zpt / 2.5)
    distratio = lumdist * 1.e6 / 10.  # lumdist is in Mpc, isochrone is computed for 10 parsecs
    return isochrone_flux * zptflux / distratio ** 2


def rebin(a, *args):
    """
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
     a=rand(6,4); b=rebin(a,3,2)
     a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)

    factor = (np.asarray(shape) / np.asarray(args)).astype('int64')

    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)]

    return eval(''.join(evList))


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        return idx - 1
    else:
        return idx


def do_find_nearest(array, vals):
    return np.array([find_nearest(array, v) for v in vals])


class _SimulateDwarfEllipticalBase:
    """
    Base class for simulating an elliptical dwarfs

    Parameters
    ----------
    npix : int
        Number of pixels for the resulting image
    oversampling : int
        Oversampling factor for internal computations
    dmin, dmax : float
        Min and max distances for galaxy in Mpc
    arcsec_per_pixel : float
        Pixel scale of image
    zpt :
        Instrument zero points that convert from Mag to counts per second in the image.
    mf_alpha : float
        The slope of the powerlaw portion of the luminosity function (-1.3 default)
    mf_mstar : float
        The mass of the break of the mass function (3.e10 solar masses defaut)
    mf_min : float
         The minimum mass (1.e5 default)
    mf_max : float
        The maximum mass
    noise_range : tuple
        Noise range
    bands : list
        List of filters to use. Default is ['g', 'r', 'i', 'z', 'y']
    auto_npix : bool
        If set true, npix may be adjusted (expanded) if the galaxy does not fit
        into the initial npix value. The max size is set by the internal
        class var max_allowed_npix
    noiseless_only : bool
        Only simulate noiseless galaxies.
    verbose : bool
        print information
    """

    instrument_name = None

    def __init__(self,
                 npix=256,
                 oversampling=5.,
                 dmin=1.,
                 dmax=10.,
                 arcsec_per_pixel=0.168,
                 zpt=27.,
                 mf_alpha=-1.3,
                 mf_mstar=3.e10,
                 mf_min=1.e5,
                 mf_max=1.e9,
                 noise_range=(0.015, 0.04),
                 bands=None,
                 auto_npix=False,
                 noiseless_only=False,
                 verbose=True):

        if bands is None:
            raise Exception("bands were not defined")

        # Input Parameters
        # ----------------
        self.oversampling = oversampling

        self.dmin = dmin
        self.dmax = dmax

        self.noise_range = noise_range

        self.bands = bands

        self.mf_alpha = mf_alpha
        self.mf_mstar = mf_mstar
        self.mf_min = mf_min
        self.mf_max = mf_max

        self.zpt = {}
        for b in bands:
            self.zpt[b] = zpt

        self.arcsec_per_pixel = arcsec_per_pixel / self.oversampling

        self.npix = npix
        self.npix_oversampled = npix * self.oversampling
        self.auto_npix = auto_npix

        self.noiseless_only = noiseless_only

        self.verbose = verbose

        # Class attribute Parameters
        # ---------------------------
        self.preferred_band = None

        # Static Parameters
        # -----------------
        # These should be redefined in the subclass __init__
        self.isochrone_dir = None
        self.isofilestring = None
        self.isochrone_column_formatter = None  # String formatter for isochrone table where the band is input
        self.inst_std = None  # Estimate of std of instrument image background (noise)
        self.max_allowed_npix = 1024  # Only used if npix is None

        # Computed Variables
        # ------------------
        self.distance = None
        self.mass = None
        self.log_mass = None
        self.isochrone = None
        self.age = None
        self.feh = None
        self.isofile = None
        self.flux_total_single_band = None

        self.redshift = None
        self.re_kpc = None
        self.re_arcsec = None
        self.re_pixels = None
        self.ellipticity = None
        self.sersic_index = None
        self.position_angle = None  # radians

        self.imf = None
        self.stochastic_mass_fraction = None
        self.minmass_stochastic = None

        self.model = None
        self.mass_model = None
        self.normalization = None
        self.model_image = None
        self.galaxy_image = None
        self.stochastic_image = None
        self.noiseless_image = None
        self.instrument_image = None

        self.x = None
        self.y = None

        self.smooth_flux = None
        self.observed_smooth_flux = None
        self.peak_smooth_flux = None

        self.psf = {}  # PSF image for each band

        self.nonzero_pm_indices = None

        self.noise_sigma = None

    def pick_galaxy(self):
        # Pick a Distance
        # ---------------
        dp = DistancePicker(dmin=self.dmin, dmax=self.dmax)
        self.distance = dp.pick_distance()

        # Pick a Mass
        # -----------
        # take a mass < 1.e9
        self.mass = self.mf_max

        while self.mass >= self.mf_max:
            self.mass = schechter_picker(1, self.mf_alpha, self.mf_mstar, self.mf_min)[0]
        self.log_mass = np.log10(self.mass)

        # Pick Isochrone
        # ---------------
        # Pick an isochrone given a mass, and compute flux at 10 pc in each band
        mp = IsochronePicker(self.isochrone_dir, self.isofilestring)
        t = mp.pick_isofile(self.log_mass, return_table=True)
        self.isochrone = mp.agerows(t)
        self.age = mp.age
        self.feh = mp.feh
        self.isofile = mp.isofile

        self._setup_isochrone()

        # Total Flux
        # ----------
        # Compute total flux for single band,
        # Prefer z filter if possible
        b = self.bands[0]
        if self.preferred_band is not None:
            b = self.preferred_band

        self.flux_total_single_band = self.mass * observed_cts_per_sec(self.smooth_flux[b],
                                                                       self.distance,
                                                                       self.zpt[b])

        # Pick a size
        sp = SizePicker()  # Returns log10 of size in pc
        self.re_kpc = 10. ** (sp.size(np.array([self.log_mass]))[0]) / 1.e3  # convert to kpc

        # Pick an axial ratio
        qpdf = AxialRatioPDF(name='qpdf')
        self.ellipticity = 1. - qpdf.rvs(size=1)[0]

        # Set up the galaxy mass model
        self._setup_galaxy()
        self._create_mass_model()

    def _setup_isochrone(self):
        isochrone = self.isochrone

        for b in self.bands:
            isochrone[b + 'flux'] = 10 ** (isochrone[self.isochrone_column_formatter.format(b)] / -2.5)

        # Select a minimum mass for the stochastic component
        self.minmass_stochastic = min(1.0, isochrone['initial_mass'][isochrone['phase'] < 0.1][-1])

        # Compute the fraction of the total mass in the stochastic component
        # (the rest will be in the smooth component)

        # self.imf = imf.Kroupa(mmin=0.05, mmax=120, p1=0.3, p2=1.3,
        #                       p3=2.3, break1=0.08, break2=0.5)

        self.imf = imf.Salpeter(mmin=0.3, mmax=120)

        self.imf.normalize()

        mmax = 200
        self.stochastic_mass_fraction = (self.imf.m_integrate(self.minmass_stochastic, mmax)[0] /
                                         self.imf.m_integrate(self.imf.mmin, mmax)[0])

        m = isochrone['initial_mass']

        dn_dm = self.imf(m)
        self.smooth_flux = {}
        selection = isochrone['initial_mass'] <= self.minmass_stochastic
        for b in self.bands:
            self.smooth_flux[b] = integrate.simps(m[selection], isochrone[b + 'flux'][selection] * dn_dm[selection])

    def _setup_galaxy(self):
        self.sersic_index = 1.0
        self.position_angle = np.random.uniform(0., np.pi)  # radians
        self.redshift = z_at_value(cosmo.luminosity_distance, self.distance * u.Mpc)
        self.re_arcsec = self.re_kpc * cosmo.arcsec_per_kpc_proper(self.redshift).value
        self.re_pixels = self.re_arcsec / self.arcsec_per_pixel

    def _create_mass_model(self):

        self.model = DwarfSersic2D(
            amplitude=1,
            r_eff=self.re_pixels,
            n=self.sersic_index,
            x_0=0, y_0=0,
            ellip=self.ellipticity,
            theta=self.position_angle)

        if self.auto_npix:

            noise_level = self.inst_std  # in the actual instrument image
            noise_level /= self.oversampling ** 2  # Down-sampled value

            max_pix = self.oversampling * self.max_allowed_npix // 2

            half_npix = self.model.compute_npix(noise_level,
                                                self.flux_total_single_band,
                                                max_pix=max_pix)

            npix = (half_npix * 2) // self.oversampling

            if npix % 2 == 1:
                npix += 1  # make npix even number

            if npix > self.npix:
                print("auto_npix resize {}->{}".format(self.npix, npix))
                self.npix = npix
                self.npix_oversampled = self.npix * self.oversampling

        # Recompute half_npix
        half_npix = self.npix_oversampled // 2

        [x, y] = np.meshgrid(np.arange(-half_npix, +half_npix, 1),
                             np.arange(-half_npix, +half_npix, 1))

        self.x = x
        self.y = y

        self.normalization = self.model.normalization
        self.model_image = self.model(x, y) / self.normalization
        self.mass_model = self.model_image * self.mass

    def renormalize_isochrone(self):
        """Renormalize the isochrone fluxes according to galaxy mass & distance"""
        for b in self.bands:
            self.isochrone[b] = observed_cts_per_sec(self.isochrone[b + 'flux'], self.distance, self.zpt[b])

    def compute_smooth_flux(self):
        """Compute the smooth flux in each band"""
        self.observed_smooth_flux = {}
        for b in self.bands:
            self.observed_smooth_flux[b] = self.mass * (
                    (1 - self.stochastic_mass_fraction) *
                    observed_cts_per_sec(self.smooth_flux[b], self.distance, self.zpt[b]))

    def create_smooth_portion(self):
        """ Create the smooth component of the galaxy images in each band"""
        self.galaxy_image = {}
        self.peak_smooth_flux = {}
        for b in self.bands:
            self.galaxy_image[b] = self.model_image * self.observed_smooth_flux[b]
            self.peak_smooth_flux[b] = self.galaxy_image[b].max()
            if self.verbose:
                print(f"Peak smooth cts/s in {b} band: {self.peak_smooth_flux[b]}")

    def create_stochastic_portion(self):
        """ Create the stochastic component of the galaxy images in each band"""

        catalog_extra = 1.1  # Draw about 1.1 times as many stars as we need to allow the model

        # cluster = imf.make_cluster(self.stochastic_mass_fraction * self.mass * catalog_extra,
        #                           massfunc='kroupa', mmin=self.minmass_stochastic)

        cluster = imf.make_cluster(self.stochastic_mass_fraction * self.mass * catalog_extra,
                                   massfunc=self.imf, mmin=self.minmass_stochastic)

        nstars = len(cluster) / catalog_extra

        # Create an image with the number of stars drawn from an appropriately normalized
        # Poisson distribution
        self._create_poisson_model(nstars)

        if self.verbose:
            print("Peak number of stars in a pixel: ", self.poisson_model.max())
            print("Pixel row,col of peak: ", np.unravel_index(self.poisson_model.argmax(),
                                                              self.poisson_model.shape))

        # For each random stochastic draw of a mass in the cluster,
        # find the nearest mass to it in the isochrone and make an array of these
        # indices into the isochrone array
        # mass_indices = np.array([(np.abs(self.isochrone['initial_mass']-m)).argmin() for m in cluster])
        mass_indices = do_find_nearest(self.isochrone['initial_mass'], cluster)

        self.stochastic_image = {}
        for b in self.bands:
            self.stochastic_image[b] = self.galaxy_image[b] * 0.

        nstars_added = 0

        pmflat = self.poisson_model.flatten()

        indices = np.nonzero(pmflat)[0]
        self.nonzero_pm_indices = indices
        npix_with_stars = len(indices)

        for idx in indices:
            iso_indices = mass_indices[nstars_added:nstars_added + pmflat[idx]]
            nstars_added += pmflat[idx]
            for b in self.bands:
                np.put(self.stochastic_image[b], idx, self.isochrone[b][iso_indices].sum())

        if self.verbose:
            print(f"{nstars_added} stars added to {npix_with_stars} pixels")

    def _create_poisson_model(self, nstars):
        # N stars is the total number of stars in the stochastic component
        # The model tells how many on average there are per pixel
        # And then we do a poisson draw on that number
        mu = nstars * self.model_image
        self.poisson_model = stats.poisson.rvs(mu, size=self.model_image.shape)

    def sum_components(self):
        if self.stochastic_image is not None:
            for b in self.bands:
                self.galaxy_image[b] += self.stochastic_image[b]

    def create_psf(self, **kwargs):
        """
        This should set the self.psf attribute which is a dictionary with a PSF image for each band.
        It should also return the self.psf dictionary
        """
        # raise NotImplementedError("create_psf needs to be overridden by inheriting subclass")
        return self.psf

    def simulate_image(self):
        """ Pick a PSF for each band, convolve, downsample, and add noise """

        # Convolve with them
        convolved_image = {}
        for b in self.bands:
            if b in self.psf.keys():
                convolved_image[b] = convolve_fft(self.galaxy_image[b], self.psf[b], allow_huge=True)
            else:
                # No PSF
                convolved_image[b] = self.galaxy_image[b]

        # Rebin to the instrument pixel scale
        self.noiseless_image = {}
        nbinnedpix = int(self.npix_oversampled / self.oversampling)
        for b in self.bands:
            self.noiseless_image[b] = rebin(convolved_image[b], nbinnedpix, nbinnedpix)

        if not self.noiseless_only:
            # Pick noise and add it to the rebinned image
            self.instrument_image = {}
            npick = NoisePicker(sigma_range=(0.015, 0.04))
            self.noise_sigma = {}
            for b in self.bands:
                self.instrument_image[b] = self.noiseless_image[b] + \
                                           npick.pick_noise(size=self.noiseless_image[b].shape)
                self.noise_sigma[b] = npick.sigma

    def _write_header_keywords(self, hdu):
        h = hdu.header
        h['logmass'] = self.log_mass
        h['distance'] = (self.distance, 'Mpc')
        h['rekpc'] = self.re_kpc
        h['rearcsec'] = self.re_arcsec
        h['FeH'] = self.feh
        h['age'] = (self.age, 'Gyr')
        h['ellip'] = (self.ellipticity, "ellipticity")
        h['pa'] = (self.position_angle, "radians")
        h['scale'] = (self.arcsec_per_pixel, "Arcsec/pixel")
        h['oversamp'] = (self.npix_oversampled, "Original oversampling")
        h['isofile'] = (os.path.basename(self.isofile), "Isochrone file")
        for b in self.bands:
            h[f'gamma_{b}'] = self.psf_gamma[b]
        for b in self.bands:
            h[f'alpha_{b}'] = self.psf_alpha[b]
        for b in self.bands:
            h[f'zpt_{b}'] = self.zpt[b]

    def save_fits(self, directory, noiseless_only=False):

        filename_template = f"d{self.distance:.2f}_"
        filename_template += f"m{self.log_mass:.2f}_re{self.re_kpc:.2f}_"
        filename_template += f"feh{self.feh:.1f}_age{self.age:.1f}"

        # Write out the noiseless image
        noiseless_file = os.path.join(directory, filename_template + "_noiseless.fits")
        noiseless_cube = np.stack([self.noiseless_image[b] for b in self.bands], axis=0)
        hdu_noiseless = fits.PrimaryHDU(noiseless_cube)
        self._write_header_keywords(hdu_noiseless)
        hdu_noiseless.writeto(noiseless_file)

        # Write out the noisy image
        if not self.noiseless_only and not noiseless_only:

            output_file = os.path.join(directory, filename_template + "{}.fits".format(self.instrument_name))
            output_cube = np.stack([self.instrument_image[b] for b in self.bands], axis=0)
            output_hdu = fits.PrimaryHDU(output_cube)
            self._write_header_keywords(output_hdu)
            for b in self.bands:
                output_hdu.header[f'noise_{b}'] = self.noise_sigma[b]
            output_hdu.writeto(output_file)

            return [noiseless_file, output_file]

        return [noiseless_file]

    def run_all_steps(self, output_directory=None, save_noiseless_only=False):

        steps = [
            self.pick_galaxy,
            self.renormalize_isochrone,
            self.compute_smooth_flux,
            self.create_smooth_portion,
            self.create_stochastic_portion,
            self.sum_components,
            self.create_psf,
            self.simulate_image
        ]

        for step in steps:
            if self.verbose:
                print(step.__name__)

            step()

        if output_directory:
            self.save_fits(output_directory, save_noiseless_only)

# ===
# HSC
# ===

class HSCDwarf(_SimulateDwarfEllipticalBase):
    """
    Simulates an elliptical dwarf using the PSFs from the HSC

    Parameters
    ----------
    npix : int
        Number of pixels for the resulting image
    oversampling : int
        Oversampling factor for internal computations
    dmin, dmax : float
        Min and max distances for galaxy in Mpc
    arcsec_per_pixel : float
        Pixel scale of image
    zpt :
        HSC zero points that convert from Mag to counts per second in the image.
    mf_alpha : float
        The slope of the powerlaw portion of the luminosity function (-1.3 default)
    mf_mstar : float
        The mass of the break of the mass function (3.e10 solar masses defaut)
    mf_min : float
         The minimum mass (1.e5 default)
    mf_max : float
        The maximum mass
    noise_range : tuple
        Noise range
    bands : list
        List of filters to use. Default is ['g', 'r', 'i', 'z', 'y']
    auto_npix : bool
        If set true, npix may be adjusted (expanded) if the galaxy does not fit
        into the initial npix value. The max size is set by the internal
        class var max_allowed_npix
    noiseless_only : bool
        Only simulate noiseless galaxies.
    verbose : bool
        print information
    """

    instrument_name = "HSC"

    def __init__(self,
                 npix=256,
                 oversampling=5.,
                 dmin=1.,
                 dmax=10.,
                 arcsec_per_pixel=0.168,
                 zpt=27.,
                 mf_alpha=-1.3,
                 mf_mstar=3.e10,
                 mf_min=1.e5,
                 mf_max=1.e9,
                 noise_range=(0.015, 0.04),
                 bands=None,
                 auto_npix=False,
                 noiseless_only=False,
                 verbose=True):

        if bands is None:
            bands = ['g', 'r', 'i', 'z', 'y']

        super().__init__(
            npix=npix,
            oversampling=oversampling,
            dmin=dmin,
            dmax=dmax,
            arcsec_per_pixel=arcsec_per_pixel,
            zpt=zpt,
            mf_alpha=mf_alpha,
            mf_mstar=mf_mstar,
            mf_min=mf_min,
            mf_max=mf_max,
            noise_range=noise_range,
            bands=bands,
            auto_npix=auto_npix,
            noiseless_only=noiseless_only,
            verbose=verbose
        )

        # Static Parameters
        # -----------------
        self.isochrone_dir = os.path.join(DATA_PATH, 'MIST_v1.2_vvcrit0.4', 'MIST_v1.2_vvcrit0.4_HSC')
        self.isofilestring = 'MIST_v1.2_feh_%s%3.2f_afe_p0.0_vvcrit0.4_HSC.iso.cmd'
        self.isochrone_column_formatter = 'hsc_{}'
        self.inst_std = 0.02 * 2  # Estimate of std of HSC image background (noise)
        self.max_allowed_npix = 1024  # Only used if npix is None

        # Other Settings
        # --------------
        self.preferred_band = 'z'

        # Computed Variables
        # ------------------
        self.psf_gamma = None
        self.psf_alpha = None

    def create_psf(self, psf_type=None, **kwargs):

        if psf_type:
            psf_downloader = HSC_PSF_DOWNLOADERS[psf_type]

            results = Table([np.array([]), np.array([]), np.array([]), np.array([])],
                            names=['band', 'gamma', 'alpha', 'fwhm'],
                            dtype=['S2', 'f4', 'f4', 'f4'])

            for band in self.bands:
                psf_path = psf_downloader(**kwargs)
                psf_fit = FitPSF(psf_path)
                psf_fit.dofit(verbose=self.verbose)
                results.add_row([band, psf_fit.gamma, psf_fit.alpha, psf_fit.fwhm * 0.17])

            self.psf_gamma = {}
            self.psf_alpha = {}
            self.psf = {}
            for line in results:
                b = line['band'].lower()

                pf = HSCMoffatPSFPicker(oversampling=self.oversampling, gamma0=line['gamma'], alpha0=line['alpha'])

                self.psf_gamma[b], self.psf_alpha[b], self.psf[b] = pf.get_oversampled_psf(verbose=self.verbose)

        else:
            # Pick PSFs
            self.psf_gamma = {}
            self.psf_alpha = {}
            self.psf = {}
            for b in self.bands:
                pf = HSCMoffatPSFPicker(oversampling=self.oversampling)
                self.psf_gamma[b], self.psf_alpha[b], self.psf[b] = pf.get_oversampled_psf(verbose=self.verbose)

        return self.psf

def simulate_hsc():
    sim = HSCDwarf()
    sim.pick_galaxy()
    sim.renormalize_isochrone()
    sim.compute_smooth_flux()
    sim.create_smooth_portion()
    sim.create_stochastic_portion()
    sim.sum_components()
    sim.simulate_image()
    return sim


# ===
# WFI
# ===

class WFIDwarf(_SimulateDwarfEllipticalBase):
    """
    Simulates an elliptical dwarf using the PSFs from the WFI on Roman S.T.

    Parameters
    ----------
    npix : int
        Number of pixels for the resulting image
    oversampling : int
        Oversampling factor for internal computations
    dmin, dmax : float
        Min and max distances for galaxy in Mpc
    arcsec_per_pixel : float
        Pixel scale of image
    zpt :
        WFI zero points that convert from Mag to counts per second in the image.
    mf_alpha : float
        The slope of the powerlaw portion of the luminosity function (-1.3 default)
    mf_mstar : float
        The mass of the break of the mass function (3.e10 solar masses defaut)
    mf_min : float
         The minimum mass (1.e5 default)
    mf_max : float
        The maximum mass
    noise_range : tuple
        Noise range
    bands : list
        List of filters to use. Default is ['R062', 'Z087', 'Y106', 'J129', 'W146', 'H158', 'F184']
    auto_npix : bool
        If set true, npix may be adjusted (expanded) if the galaxy does not fit
        into the initial npix value. The max size is set by the internal
        class var max_allowed_npix
    noiseless_only : bool
        Only simulate noiseless galaxies.
    verbose : bool
        print information
    """

    instrument_name = "WFI"

    def __init__(self,
                 npix=256,
                 oversampling=5.,
                 dmin=1.,
                 dmax=10.,
                 arcsec_per_pixel=0.168,
                 zpt=27.,
                 mf_alpha=-1.3,
                 mf_mstar=3.e10,
                 mf_min=1.e5,
                 mf_max=1.e9,
                 noise_range=(0.015, 0.04),
                 bands=None,
                 auto_npix=False,
                 noiseless_only=False,
                 verbose=True):

        if bands is None:
            bands = ['R062', 'Z087', 'Y106', 'J129', 'W146', 'H158', 'F184']

        super().__init__(
            npix=npix,
            oversampling=oversampling,
            dmin=dmin,
            dmax=dmax,
            arcsec_per_pixel=arcsec_per_pixel,
            zpt=zpt,
            mf_alpha=mf_alpha,
            mf_mstar=mf_mstar,
            mf_min=mf_min,
            mf_max=mf_max,
            noise_range=noise_range,
            bands=bands,
            auto_npix=auto_npix,
            noiseless_only=noiseless_only,
            verbose=verbose
        )

        # Static Parameters
        # -----------------
        self.isochrone_dir = os.path.join(DATA_PATH, 'MIST_v1.2_vvcrit0.4_WFIRST')
        self.isofilestring = 'MIST_v1.2_feh_%s%3.2f_afe_p0.0_vvcrit0.4_WFIRST.iso.cmd'
        self.isochrone_column_formatter = '{}'
        self.inst_std = 0.053649535982253485 # Estimate of std of WFI image background (noise)
        self.max_allowed_npix = 1024  # Only used if npix is None

        # Other Settings
        # --------------
        self.preferred_band = 'Y106'


    def create_psf(self, **kwargs):

        for b in self.bands:
            wfi = WFI()
            wfi.filter = b.replace(b[0], 'F')
            psf_hdul = wfi.calc_psf(oversample=int(self.oversampling))
            self.psf[b] = psf_hdul[0].data

        return self.psf


def simulate_wfi():
    sim = WFIDwarf()
    sim.pick_galaxy()
    sim.renormalize_isochrone()
    sim.compute_smooth_flux()
    sim.create_smooth_portion()
    sim.create_stochastic_portion()
    sim.sum_components()
    sim.simulate_image()
    return sim


