import numpy as np
import make_galaxies_grid_mtot
import sys

def dist_mass(logmass):
  for d in np.arange(4.0,16.,0.25):
        print(f"Galaxy ------------------------------------------------------------------")
        sim = make_galaxies_grid_mtot.simulate_HSC(distance=d,logmass=logmass,age=11.) 
        sim.save_fits("grid2")

if __name__ == "__main__":
    logmass = float(sys.argv[1])
    dist_mass(logmass)
