from dsii_dwarfs.simulate_psf_fit_dwarf_elliptical import SimulatePSFFitDwarfElliptical

def test_simulate_psf_fit_dwarf_elliptical():
    sim = SimulatePSFFitDwarfElliptical()
    sim.run_all_steps()