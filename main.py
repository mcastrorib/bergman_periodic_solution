import numpy as np
from periodicPoreArray import PeriodicPoreArray

if __name__ == "__main__":
        
    # --- Essential parameters
    # -- [LENGTH]   => um
    # -- [TIME]     => ms
    unit_cell_length = 10.0
    unit_cell_divisions = 1
    unit_cell_spatial_divisions = 10
    unit_cell_reciprocal_divisions = 2
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
    wavevector_K = np.array([0.0, 0.0, 0.0])
    lattice.solve(wavevector_K)

    # Visualize data
    lattice.dataviz_theta()
        
    # Debug
    theta_diff = np.asmatrix(np.zeros([lattice.get_points(), lattice.get_points()]))
    for row in range(lattice.get_points()):
        for col in range(lattice.get_points()):
            theta_diff[row, col] = lattice.theta_Mg[row, col] + lattice.theta_Pg[row, col]
    
    print("theta_Mg + theta_Pg = ")
    for row in range(lattice.get_points()):
        print(theta_diff[row])
