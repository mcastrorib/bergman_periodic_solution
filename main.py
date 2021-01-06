import numpy as np
from periodicPoreArray import PeriodicPoreArray

if __name__ == "__main__":
        
    # --- Essential parameters
    # -- [LENGTH]   => um
    # -- [TIME]     => ms
    unit_cell_length = 10.0
    unit_cell_divisions = 4
    unit_cell_spatial_divisions = 10
    unit_cell_reciprocal_divisions = unit_cell_divisions
    grain_radius = 5.0
    D_p = 2.5

    # Create pore lattice object
    lattice = PeriodicPoreArray()
    lattice.set_unit_cell_length(unit_cell_length)
    lattice.set_unit_cell_divisions(unit_cell_divisions)
    lattice.set_grain_radius(grain_radius)
    lattice.set_fluid_diffusion_coefficient(D_p)
    lattice.set_spatial_divisions(unit_cell_spatial_divisions)
    lattice.set_reciprocal_divisions(unit_cell_reciprocal_divisions)
    lattice.build()
 
    # Solve
    time = 10 *(unit_cell_length**2 / D_p) # time is in miliseconds
    K = np.array([1.0, 0.0, 0.0])
    lattice.set_k_points(50)
    lattice.set_ka_max(15.0)
    lattice.solve(K, time)

    # Visualize data
    # lattice.dataviz_theta()
    lattice.dataviz_Mkt()
        
