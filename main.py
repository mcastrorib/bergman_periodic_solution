import numpy as np
from periodicPoreArray import PeriodicPoreArray

if __name__ == "__main__":
        
    # --- Essential parameters
    # -- [LENGTH] = um
    # -- [TIME] = ms
    unit_cell_length = 1.0
    unit_cell_divisions = 1
    grain_radius = 0.5
    D_p = 2.5

    # Create pore lattice object
    pore_lattice = PeriodicPoreArray()
    pore_lattice.set_unit_cell_length(unit_cell_length)
    pore_lattice.set_unit_cell_divisions(unit_cell_divisions)
    pore_lattice.set_grain_radius(grain_radius)
    pore_lattice.set_fluid_diffusion_coefficient(D_p)
    pore_lattice.build()

    # Visualize data
    pore_lattice.dataviz_theta()
