import numpy as np
from make_galaxies import SimulateRandomDwarfElliptical,simulate_HSC
import make_galaxies

for i in range(2000):
    print(f"Galaxy {i} ------------------------------------------------------------------")
    sim = SimulateRandomDwarfElliptical(dmax=15.,MF_min=1.e5,MF_max=1.e9)
    sim.pick_galaxy()
    print("mass: ",np.log10(sim.mass))
    print("distance: ",sim.distance)
    print("re_kpc, arcsec: ",sim.re_kpc, sim.re_arcsec)
    print("[Fe/H]: ",sim.feh)
    print("age: ",sim.age)
    print("ellip: ",sim.ellipticity)
    print("pa: ",sim.position_angle)
    print("")

    sim.renormalize_isochrone()
    sim.compute_smooth_flux()
    sim.create_smooth_portion()

    sim.create_stochastic_portion()
    sim.sum_components()
    print("")
    print("HSC image")
    sim.simulate_HSCimage()
    sim.save_fits('stamps2')


