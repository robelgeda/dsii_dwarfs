import numpy as np
import make_galaxies_grid

def dist_mass():
  for d in [2.,4.,8.,16.]:
    for m in [5.5,6.5,7.5]:
        print(f"Galaxy ------------------------------------------------------------------")
        sim = make_galaxies_grid.SimulateDwarfElliptical(npix=256,
                  distance=d,logmass=m, age=12., feh=-1.75)
        print("mass: ",np.log10(sim.mass))
        print("distance: ",sim.distance)
        print("re_kpc, arcsec: ",sim.re_kpc, sim.re_arcsec)
        print("[Fe/H]: ",sim.feh)
        print("age: ",sim.age/1.e9)
        print("ellip: ",sim.ellipticity)
        print("pa: ",sim.position_angle)
        sim.renormalize_isochrone()
        sim.compute_smooth_flux()
        sim.create_smooth_portion()
        sim.create_stochastic_portion()
        sim.sum_components()
        print("")
        print("HSC image")
        sim.simulate_HSCimage()
        sim.save_fits("grid2")

def age_metallicity():
  d=10
# m = 6
  m = 7
  for age in [3.,7.,10.,13.]:
    for feh in [-2.,-1.5,-1.]:
        print(f"Galaxy ------------------------------------------------------------------")
        sim = make_galaxies_grid.SimulateDwarfElliptical(npix=256,
                  distance=d,logmass=m, age=age, feh=feh)
        print("mass: ",np.log10(sim.mass))
        print("distance: ",sim.distance)
        print("re_kpc, arcsec: ",sim.re_kpc, sim.re_arcsec)
        print("[Fe/H]: ",sim.feh)
        print("age: ",sim.age/1.e9)
        print("ellip: ",sim.ellipticity)
        print("pa: ",sim.position_angle)
        sim.renormalize_isochrone()
        sim.compute_smooth_flux()
        sim.create_smooth_portion()
        sim.create_stochastic_portion()
        sim.sum_components()
        print("")
        print("HSC image")
        sim.simulate_HSCimage()
        sim.save_fits("grid2")

def age_metallicity_fix():
  d=10
# m = 6
  m = 7
  for age in [7.]:
    for feh in [-2.]:
        print(f"Galaxy ------------------------------------------------------------------")
        sim = make_galaxies_grid.SimulateDwarfElliptical(npix=256,
                  distance=d,logmass=m, age=age, feh=feh)
        print("mass: ",np.log10(sim.mass))
        print("distance: ",sim.distance)
        print("re_kpc, arcsec: ",sim.re_kpc, sim.re_arcsec)
        print("[Fe/H]: ",sim.feh)
        print("age: ",sim.age/1.e9)
        print("ellip: ",sim.ellipticity)
        print("pa: ",sim.position_angle)
        sim.renormalize_isochrone()
        sim.compute_smooth_flux()
        sim.create_smooth_portion()
        sim.create_stochastic_portion()
        sim.sum_components()
        print("")
        print("HSC image")
        sim.simulate_HSCimage()
        sim.save_fits("grid2")

if __name__ == "__main__":
#   dist_mass()
#   age_metallicity()
    age_metallicity_fix()
