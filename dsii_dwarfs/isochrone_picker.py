import os
import numpy as np
from scipy import stats
from astropy.table import Table

from . import DATA_PATH

class IsochronePicker:
    """ 
    Choose the isochrone that is the closest match in age and metallicity
    Return the rows that correspond to the selected age.
    Optionally:
    Draw metallicities at random from the Kirby+13 Mass-Metallicity relation
    Draw ages at random from a t distribution with 5 degrees of freedom 
    
    See validation in Isochrone_picker.ipynb
    """

    def __init__(self,
           isochrone_dir=os.path.join(DATA_PATH, 'MIST_v1.2_vvcrit0.4', 'MIST_v1.2_vvcrit0.4_HSC'),
           isofilestring='MIST_v1.2_feh_%s%3.2f_afe_p0.0_vvcrit0.4_HSC.iso.cmd',
           mean_age=11., age_sigma=2., # Gyr
           noscatter=False):

        # For making grids, remove the scatter in the mean relations of mass-metallicity and age
        self.noscatter=noscatter

        # Mass-metallicity relation from Kirby et al. 2013 (2013 ApJ 779 102)
        self.c0 = -1.69
        self.c1 = 0.30
        self.sigma = 0.1

        # MIST isochrones
        self.metallicities = np.array(
            [-4.,-3.5,-3.,-2.5,-2.,-1.75,-1.5,-1.25,-1.,-0.75,-0.5,-0.25,0.,0.25,0.5])

        self.isodir = isochrone_dir
        self.isofilestring = isofilestring

        # Use a t distribution with 5 degrees of freedom for scatter about the mean relation
        # This gives a few more outliers than a Gaussian
        self.Zscatter = stats.t(loc=0,scale=self.sigma,df=5)

        # Ages just assume t distribution with a mean & scatter
        self.meanage = mean_age
        self.agesigma = age_sigma
        self.agescatter = stats.norm(loc=self.meanage,scale=self.agesigma)

        self.age = None
        self.isofile = None
        self.feh = None

    def pick_metallicity(self, logmass):
        if self.noscatter:
            feh = self.c1*(logmass-6)+self.c0
        else:
            feh = self.c1*(logmass-6)+self.c0 + self.Zscatter.rvs(len(logmass))
        return feh
            
    def pick_age(self, age_gyr = 13, max_age_gyr = 15):
        if self.noscatter:
            age = age_gyr
        else:
            age = max_age_gyr
            while age >= max_age_gyr:
                age = self.agescatter.rvs(1)[0] # Gyr
        return 1.e9*age # Return age in years

    def pick_isofile(self, logmass, return_table=False):
        metallicity = self.pick_metallicity(np.array([logmass]))[0] # get back a float
        i = np.array(np.abs(metallicity-self.metallicities)).argmin()

        feh = self.metallicities[i]

        if feh < 0:
            self.isofile = os.path.join(self.isodir, self.isofilestring % ('m',-feh))
        else:
            self.isofile = os.path.join(self.isodir, self.isofilestring % ('p',feh))

        self.feh = feh

        if return_table:
            t = Table.read(self.isofile,format='ascii.commented_header',header_start=-1)
            return t
        else:
            return self.isofile

    def agerows(self, isotable, age_gyr=13.):
        age = self.pick_age(age_gyr=age_gyr)
        logage = np.log10(age)
        i = np.array(np.abs(logage-isotable['log10_isochrone_age_yr'])).argmin()
        selected_age = isotable['log10_isochrone_age_yr'][i]
        tt = isotable[np.isclose(isotable['log10_isochrone_age_yr'],selected_age)]
        self.age = 10.**logage/1.e9 # Gyr
        return tt
