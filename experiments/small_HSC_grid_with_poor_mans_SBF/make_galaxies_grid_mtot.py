# Create a bunch of fake dE galaxies
# With a specific set of parameters rather than random
import os

import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.convolution import convolve_fft
from scipy import integrate, stats
from astropy import modeling
from astropy.table import Table
from astropy.io import fits

# For integrating sersic profile
from scipy.special import gamma, gammaincinv, gammainc

import imf # https://github.com/keflavich/imf

from dsii_dwarfs.axial_ratio_picker import AxialRatioPDF
from dsii_dwarfs.distance_picker import DistancePicker
from dsii_dwarfs.isochrone_picker import IsochronePicker
from dsii_dwarfs.noise_picker import NoisePicker
from dsii_dwarfs.psf_picker import HSCMoffatPSFPicker
from dsii_dwarfs.schechter_picker import schechter_picker
#from dsii_dwarfs.size_picker import SizePicker
from size_picker import SizePicker

def sersic_integral(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = gammaincinv(2*n, 0.5)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n

class SimulateDwarfElliptical:
    def __init__(self,noise_rms=0.02,
                 arcsec_per_pixel = 0.168, oversampling=5., 
                 bands = ['g','r','i','z','y'], zpt = 27.,
                 npix=256,distance=10.,logmass=7.,age=11.):
        self.noise_rms = noise_rms
        self.bands = bands
        self.oversampling = oversampling
        self.zpt = {}
        for b in bands:
            self.zpt[b] = zpt
        self.arcsec_per_pixel = arcsec_per_pixel/self.oversampling 
        self.Nospix = npix*self.oversampling
        self.distance,self.logmass,self.age = distance,logmass,age
        self.mass = 10.**self.logmass
        self.select_isochrone(age_gyr=age,noscatter=True)
        self.pick_morphology()

    def info(self):
        print(f"logmass,distance:         {self.logmass:.2f} {self.distance:.2f}")
        print(f"re (kpc,arcsec,pixels):   {self.re_kpc:.2f} {self.re_arcsec:.2f} {self.re_pixels:.1f}")
        print(f"ellipticity, pa:          {self.ellipticity:.2f}, {self.position_angle*180/np.pi:.0f}")
        print(f"age (Gyr),[Fe/H]:         {self.age:.2f}, {self.feh:.2f}")
        print(f"Mass fraction in image:   {self.mass_fraction_in_image:.3f}")
        print(f"Stochastic Min_mass frac: {self.minmass_stochastic:.2f},{self.stochastic_mass_fraction:.2f}")
        print("Peak number of stars in a pixel: ",self.poisson_model.max())
        print("Pixel row,col of peak: ",np.unravel_index(self.poisson_model.argmax(),
              self.poisson_model.shape))

        print(f"{self.isofile}")
        t = Table()
        print("")
        t['band'] = self.bands
        t['ABmag'] = [self.total_mag[b] for b in self.bands]
        t['smooth_flux'] = [self.observed_smooth_flux[b] for b in self.bands]
        t['peak_smooth_flux'] = [self.peak_smooth_flux[b] for b in self.bands]
        t['psf_gamma'] = [self.psf_gamma[b] for b in self.bands]
        t['psf_alpha'] = [self.psf_alpha[b] for b in self.bands]
        t['ABmag'].format = '.2f'
        t['smooth_flux'].format = '.2f'
        t['peak_smooth_flux'].format = '.6f'
        t['psf_gamma'].format = '.2f'
        t['psf_alpha'].format = '.2f'
        print(t)

    def select_isochrone(self,age_gyr=11.,noscatter=True):
        """ Pick an isochrone of selected age and mass """
        mp = IsochronePicker(noscatter=noscatter,mean_age=age_gyr)
        isotable = mp.pick_isofile(self.logmass,return_table=True)
        self.isofile = mp.isofile
        self.isochrone = mp.agerows(isotable, age_gyr=age_gyr)
        self.age =  mp.age
        self.feh =  mp.feh
        self.setup_isochrone()

    def pick_morphology(self):
        # Pick a size
        sp = SizePicker(noscatter=True) # Returns log10 of size in pc
        self.re_kpc = 10.**(sp.size(np.array([self.logmass]))[0]) / 1.e3 # convert to kpc
        # Pick an axial ratio
        qpdf = AxialRatioPDF(name='qpdf')
        # self.ellipticity = 1.-qpdf.rvs(size=1)[0]  
        self.ellipticity = 0. # Make them all round
        # Set up the galaxy mass model
        self.setup_galaxy()
        self.create_mass_model()

    def renormalize_isochrone(self):
        """Renormalize the isochrone fluxes according to galaxy mass & distance"""
        for b in self.bands:
            self.isochrone[b] = observed_cts_per_sec(self.isochrone[b+'flux'],self.distance,self.zpt[b])

    def compute_smooth_flux(self):
        """Compute the smooth flux in each band"""
        self.observed_smooth_flux = {}
        for b in self.bands:
            self.observed_smooth_flux[b] = self.mass * (
                (1-self.stochastic_mass_fraction) * 
                observed_cts_per_sec(self.smooth_flux[b],self.distance,self.zpt[b]))

    def create_smooth_portion(self):
        """Create the smooth component of the galaxy images in each band"""
        self.galaxy_image = {}
        self.peak_smooth_flux = {}
        for b in self.bands:
            self.galaxy_image[b]=self.model_image*self.observed_smooth_flux[b]
            self.peak_smooth_flux[b] = self.galaxy_image[b].max()
#           print(f"Peak smooth cts/s in {b} band: {self.peak_smooth_flux[b]}")

    def create_stochastic_portion(self):
        """ Create the stochastic component of the galaxy images in each band"""
        catalog_extra = 1.1 # Draw about 1.1 times as many stars as we need to allow the model
        # Kroupa is not working with mmin; changing to Salpeter
        # cluster = imf.make_cluster(self.stochastic_mass_fraction*self.mass*catalog_extra,
        #                   massfunc='kroupa',mmin=self.minmass_stochastic)
        cluster = imf.make_cluster(self.stochastic_mass_fraction*self.mass*catalog_extra,
                           massfunc=self.imf,mmin=self.minmass_stochastic)
        nstars = len(cluster)/catalog_extra
        # Create an image with the number of stars drawn from an appropriately normalized
        # Poisson distribution
        self.create_poisson_model(nstars)
#       print("Peak number of stars in a pixel: ",self.poisson_model.max())
#       print("Pixel row,col of peak: ",np.unravel_index(self.poisson_model.argmax(),
#             self.poisson_model.shape))
        # For each random stochastic draw of a mass in the cluster, 
        # find the nearest mass to it in the isochrone and make an array of these 
        # indices into the isochrone array
        #mass_indices = np.array([(np.abs(self.isochrone['initial_mass']-m)).argmin() for m in cluster])
        mass_indices = do_find_nearest(self.isochrone['initial_mass'],cluster)
        self.stochastic_image = {}
        for b in self.bands:
            self.stochastic_image[b]=self.galaxy_image[b]*0.
        xint,yint = self.x.flatten().astype(np.int32),self.y.flatten().astype(np.int32)
        nstars_added = 0
        pmflat = self.poisson_model.flatten()
        indices = np.nonzero(pmflat)[0]
        self.nonzero_pm_indices = indices
        npix_withstars = len(indices)
        for idx in indices:
            iso_indices = mass_indices[nstars_added:nstars_added+pmflat[idx]]
            nstars_added += pmflat[idx]
            for b in self.bands:
                np.put(self.stochastic_image[b],idx,self.isochrone[b][iso_indices].sum())
        print(f"{nstars_added} stars added to {npix_withstars} pixels")

#       for xx,yy in zip(xint,yint):
#           if self.poisson_model[xx,yy] > 0:
#               npix_withstars += 1
#               nstars_added += self.poisson_model[xx,yy]
#               random_indices = np.random.randint(0,len(mass_indices),size=self.poisson_model[xx,yy])
#               for b in self.bands:
#                   self.stochastic_image[b][xx,yy] += self.isochrone[b][mass_indices[random_indices]].sum()

    def sum_components(self):
        for b in self.bands:
            self.galaxy_image[b] += self.stochastic_image[b]

    def simulate_HSCimage(self,psfgamma=2.5,psfsigma_log_gamma=0.001,noiserange=(0.02,0.02)):
        """ Pick a PSF for each band, convolve, downsample, and add noise """
        # Pick PSFs
        self.psf_gamma = {}
        self.psf_alpha = {}
        self.psf = {}
        for b in self.bands:
            pf = HSCMoffatPSFPicker(oversampling=self.oversampling,gamma0=psfgamma,sigma_log_gamma=psfsigma_log_gamma)
            self.psf_gamma[b],self.psf_alpha[b],self.psf[b] = pf.get_oversampled_psf(verbose=False)
        # Convolve with them
        convolved_image = {}
        for b in self.bands:
            convolved_image[b] = convolve_fft(self.galaxy_image[b],self.psf[b])
        # Rebin to the HSC pixel scale
        self.noiseless_image = {}
        nbinnedpix = int(self.Nospix/self.oversampling)
        for b in self.bands:
            self.noiseless_image[b] = rebin(convolved_image[b],nbinnedpix,nbinnedpix)
        # Pick noise and add it to the rebinned image
        self.HSCimage = {}
        npick = NoisePicker(sigma_range=noiserange)
        self.noise_sigma = {}
        for b in self.bands:
            self.HSCimage[b] = self.noiseless_image[b] + \
                 npick.pick_noise(size=self.noiseless_image[b].shape)
            self.noise_sigma[b] = npick.sigma

    def setup_isochrone(self):
        isochrone=self.isochrone
        for b in self.bands:
            isochrone[b+'flux'] = 10**(isochrone['hsc_'+b]/-2.5)
        # Select a minimum mass for the stochastic component
        self.minmass_stochastic = min(1.0,isochrone['initial_mass'][isochrone['phase']<0.1][-1])
        # Compute the fraction of the total mass in the stochastic component (the rest will be in the smooth component)
        # 9/10/20 HF Changing the minimum mass from 0.1 to 0.03....might need to calibrate mass-luminosity relation?
        # 9/11/20 HF Changing Salpeter to avoid mmin bug in imf.py; might need to calibrate M-L relation
        # self.imf = imf.Kroupa(mmin=0.03, mmax=120, p1=0.3, p2=1.3, p3=2.3,
        #          break1=0.08, break2=0.5)
        self.imf = imf.Salpeter(mmin=0.3, mmax=120)
        self.imf.normalize()
        mmax = 120
        self.stochastic_mass_fraction = (self.imf.m_integrate(self.minmass_stochastic, mmax)[0] /
                 self.imf.m_integrate(self.imf.mmin, mmax)[0])
        m = isochrone['initial_mass']
        dn_dm = self.imf(m)
        self.smooth_flux = {}
        selection = isochrone['initial_mass'] <= self.minmass_stochastic
        for b in self.bands:
            # note argument order simps(y,x) 
            self.smooth_flux[b] = integrate.simps(isochrone[b+'flux'][selection]*dn_dm[selection],m[selection]) 

    def setup_galaxy(self):
        self.sersic_index = 1.0
        self.position_angle = np.random.uniform(0.,np.pi) # radians
        self.redshift = z_at_value(cosmo.luminosity_distance,self.distance*u.Mpc)
        self.re_arcsec = self.re_kpc*cosmo.arcsec_per_kpc_proper(self.redshift).value
        self.re_pixels = self.re_arcsec/self.arcsec_per_pixel
        
    def create_mass_model(self):
        N=self.Nospix
        x,y = np.meshgrid(np.arange(N),np.arange(N))
        self.x = x
        self.y = y
        self.model = modeling.models.Sersic2D(
            amplitude=1,
            r_eff=self.re_pixels,
            n=self.sersic_index,
            x_0=N/2,y_0=N/2,
            ellip=self.ellipticity,theta=self.position_angle)
        mass_in_image = np.sum(self.model(x,y).flat)
        total_mass = sersic_integral(1.,self.re_pixels,self.sersic_index)
        self.mass_fraction_in_image = mass_in_image/total_mass
        self.normalization = total_mass
        self.model_image = self.model(x,y)/self.normalization
        self.mass_model = self.model_image*self.mass
        
    def calculate_total_magnitudes(self,zpt=27.):
        """ Calculate the total magnitudes in each band """
        # Inputs to the calculation:
        #  - Noiseless images (these are in counts/second)
        #  - Fraction of the mass enclosed in the image
        #  - Zeropoints -- for converting back to flux and AB mag
        #  flux = sum(noiseless_image)/fraction_of_mass 
        self.total_mag = {} 
        for b in self.bands:
            total_cts_per_sec = self.galaxy_image[b].sum() / self.mass_fraction_in_image
            self.total_mag[b] = -2.5*np.log10(total_cts_per_sec) + zpt

    def create_poisson_model(self,nstars):
        # N stars is the total number of stars in the stochastic component
        # The model tells how many on average there are per pixel
        # And then we do a poisson draw on that number
        mu = nstars*self.model_image
        self.poisson_model = stats.poisson.rvs(mu,size=self.model_image.shape) 

    def save_fits(self,directory):
        filename_template = f"d{self.distance:.2f}_"
        filename_template += f"m{self.logmass:.2f}_re{self.re_kpc:.2f}_"
        filename_template += f"feh{self.feh:.1f}_age{self.age:.1f}"
        # Write out the noiseless image
        noiseless_file = directory+"/"+filename_template+"_noiseless.fits"
        noiseless_cube = np.stack([self.noiseless_image[b] for b in self.bands],axis=0)
        hdu_noiseless = fits.PrimaryHDU(noiseless_cube)
        self.write_header_keywords(hdu_noiseless)
        hdu_noiseless.writeto(noiseless_file)
        # Write out the HSC noisy image
        HSC_file = directory+"/"+filename_template+"_HSC.fits"
        HSC_cube = np.stack([self.HSCimage[b] for b in self.bands],axis=0)
        hdu_HSC = fits.PrimaryHDU(HSC_cube)
        self.write_header_keywords(hdu_HSC)
        for b in self.bands:
           hdu_HSC.header[f'noise_{b}'] = self.noise_sigma[b]
        hdu_HSC.writeto(HSC_file)

    def write_header_keywords(self,hdu):
        h = hdu.header
        h['logmass'] = self.logmass
        h['distance'] = (self.distance, 'Mpc')
        h['rekpc'] = self.re_kpc
        h['rearcsec'] = self.re_arcsec
        h['FeH'] = self.feh
        h['age'] = (self.age, 'Gyr')
        h['ellip'] = (self.ellipticity, "ellipticity")
        h['pa'] = (self.position_angle, "radians")
        h['scale'] = (self.arcsec_per_pixel*self.oversampling, "Arcsec/pixel")
        h['oversamp'] = (self.Nospix, "Original oversampling")
        h['isofile'] = (os.path.basename(self.isofile), "Isochrone file")
        for b in self.bands:
           h[f'ABmag_{b}'] = self.total_mag[b]
        for b in self.bands:
           h[f'gamma_{b}'] = self.psf_gamma[b]
        for b in self.bands:
           h[f'alpha_{b}'] = self.psf_alpha[b]
        for b in self.bands:
           h[f'zpt_{b}'] = self.zpt[b]

def observed_cts_per_sec(isochrone_flux,lumdist,zpt=27.0):
    zptflux = 10.**(zpt/2.5)
    distratio = lumdist *1.e6/10.  # lumdist is in Mpc, isochrone is computed for 10 parsecs
    return isochrone_flux*zptflux/distratio**2

def rebin(a, *args):
    """
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = (np.asarray(shape)/np.asarray(args)).astype('int64')
#   print(np.asarray(shape).dtype,np.asarray(args).dtype,factor.dtype)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] \
#             + ['/factor[%d]'%i for i in range(lenShape)]
#   print(''.join(evList))
    return eval(''.join(evList))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx
def do_find_nearest(array,vals):
    return np.array([find_nearest(array,v) for v in vals])

def simulate_HSC(distance,logmass,age):
    sim = SimulateDwarfElliptical(npix=256,distance=distance,logmass=logmass,age=age)
    sim.renormalize_isochrone()
    sim.compute_smooth_flux()
    sim.create_smooth_portion()
    sim.create_stochastic_portion()
    sim.sum_components()
    sim.calculate_total_magnitudes()
    sim.simulate_HSCimage()
    sim.info()
    return sim
