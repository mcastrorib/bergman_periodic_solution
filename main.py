import numpy as np
from periodicPoreArray import PeriodicPoreArray

if __name__ == "__main__":
        
    # --- Essential parameters
    # -- [LENGTH]   => um
    # -- [TIME]     => ms
    unit_cell_length = 10.0
    unit_cell_divisions = 1
    grain_radius = 5.0
    D_p = 2.5

    # Create pore lattice object
    lattice = PeriodicPoreArray()
    lattice.set_unit_cell_length(unit_cell_length)
    lattice.set_unit_cell_divisions(unit_cell_divisions)
    lattice.set_grain_radius(grain_radius)
    lattice.set_fluid_diffusion_coefficient(D_p)
    lattice.build()

    # Visualize data
    lattice.dataviz_theta()
 
    # Solve
    wavevector_K = np.array([0.0, 0.0, 0.0])
    lattice.solve(wavevector_K) 


        

