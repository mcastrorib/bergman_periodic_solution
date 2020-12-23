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
 
    # Solve
    wavevector_K = np.array([0.0, 0.0, 0.0])
    lattice.solve(wavevector_K)

    # -- Debug
    # check if theta_Pg is symmetric
    symmat = np.asmatrix(np.zeros([lattice.get_points(), lattice.get_points()]))
    for row in range(lattice.get_points()):
        for col in range(lattice.get_points()):
            symmat[row, col] = np.real(lattice.get_theta_Pg(row, col) - lattice.get_theta_Pg(col, row)) 

    print("symmetry of theta_Pg:")     
    for row in range(lattice.get_points()):
        print(symmat[row])
    
    print("diagonal of theta_Pg:")
    for row in range(lattice.get_points()):
        print("theta_Pr[{}] = {}".format(row, np.real(lattice.get_theta_Pg(row, row))))

    # Visualize data
    lattice.dataviz_theta()
        

