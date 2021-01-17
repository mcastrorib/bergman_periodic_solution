import numpy as np
from periodicPoreArray import PeriodicPoreArray
from LeastSquaresRegression import LeastSquaresRegression as lsr

if __name__ == "__main__":
        
    # --- Essential parameters
    # -- [LENGTH]   => um
    # -- [TIME]     => ms
    unit_cell_length = 10.0
    unit_cell_divisions = 1
    unit_cell_spatial_divisions = 5
    unit_cell_reciprocal_divisions = unit_cell_divisions
    grain_radius = 5.0
    D_p = 2.5
    solve_Mkt = False
    solve_Dt = True

    # Create pore lattice object
    lattice = PeriodicPoreArray()
    lattice.set_unit_cell_length(unit_cell_length)
    lattice.set_unit_cell_divisions(unit_cell_divisions)
    lattice.set_grain_radius(grain_radius)
    lattice.set_fluid_diffusion_coefficient(D_p)
    lattice.set_spatial_divisions(unit_cell_spatial_divisions)
    lattice.set_reciprocal_divisions(unit_cell_reciprocal_divisions)
    lattice.build()
 
    # Solve M(k,t)
    if(solve_Mkt):
        times = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] # 1 *(unit_cell_length**2 / D_p) # time is in miliseconds
        K = np.array([1.0, 0.0, 0.0])
        lattice.set_k_points(50)
        lattice.set_ka_max(5*np.pi)
        Mkt_data = []
        Mkt_normdata = []  
        for time in times:
            # solve problem
            print("Solving M(k,t) for t = {:.2f} ms".format(time))
            lattice.solve(K, time)
            
            # get raw data
            Mkt_data.append(np.real(lattice.Mkt.copy()))

            # get normalized data
            lattice.normalize_Mkt()
            Mkt_normdata.append(np.real(lattice.Mkt.copy()))

        # Visualize data
        # lattice.dataviz_theta()
        labels = [str(time) + "ms" for time in times] 
        lattice.dataviz_Mkts(Mkt_data, labels)
        lattice.dataviz_Mkts(Mkt_normdata, labels)
    
    # if(solve_Dt):
    # Solve D(t)/Dp
    time_scale = (unit_cell_length**2 / D_p) # time is in miliseconds
    times = np.linspace(0.1, 2.5, 10)
    # times = np.array([0.1])
    times = time_scale * times
    K = np.array([1.0, 0.0, 0.0])
    lattice.set_k_points(4)
    lattice.set_ka_max(0.1)
    Mkt_data = []
    Mkt_normdata = []  
    Dt_data = []
    Dt_ls_data = []
    for time in times:
        # solve problem
        print("Solving M(k,t) for t = {:.2f} ms".format(time))
        lattice.solve(K, time)
        lattice.normalize_Mkt()
        M0t = lattice.get_Mkt(-1)
        k0 = lattice.get_wavevector_k(-1)
        lattice.solve_Dt(M0t, k0, time)            
        
        # get raw data
        Dt_data.append(lattice.get_Dt())

        # get LSR data
        rhs = [((-1.0) * lattice.get_fluid_diffusion_coefficient() * time * np.linalg.norm(wk)**2) for wk in lattice.wavevector_k]
        lhs = [np.log(np.real(Mk)) for Mk in lattice.Mkt]
        my_ls = lsr()
        my_ls.config(rhs, lhs, 4)
        my_ls.solve()
        Dt_ls_data.append(my_ls.results()["B"])

        # get raw data
        Mkt_data.append(np.real(lattice.Mkt.copy()))

        # get normalized data
        lattice.normalize_Mkt()
        Mkt_normdata.append(np.real(lattice.Mkt.copy()))
    
    # Visualize data 
    labels = [str(time) + "ms" for time in times] 
    lattice.dataviz_Mkts(Mkt_data, labels)
    lattice.dataviz_Mkts(Mkt_normdata, labels)
    
    lattice.dataviz_Dts(Dt_data, times)
    lattice.dataviz_Dts(Dt_ls_data, times)
