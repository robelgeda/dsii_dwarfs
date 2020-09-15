from ..simulate_dwarf_elliptical import HSCDwarf, WFIDwarf

def test_hsc_dwarf(*args, **kwargs):
    hsc_sim = HSCDwarf()
    hsc_sim.run_all_steps()


def test_wfi_dwarf(*args, **kwargs):
    wfi_sim = WFIDwarf()
    wfi_sim.run_all_steps()
