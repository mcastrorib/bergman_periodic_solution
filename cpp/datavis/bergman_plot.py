import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def read_parameters_file(filepath):
    N = 0
    Dp = 0.0
    Dm = 0.0
    cell_length = 0.0
    sphere_radius = 0.0
    rho = 0.0

    # read data from file
    with open(filepath, 'r') as txt_file:
        # ignore header line
        next(txt_file)

        # split data with comma separator
        line = txt_file.readline().split(',')

        # process data
        N = int(line[0])
        Dp = float(line[1])
        Dm = float(line[2])
        cell_length = float(line[3])
        sphere_radius = float(line[4])
        rho = float(line[5])

    Params_data = {
        'N': N,
        'Dp': Dp,
        'Dm': Dm,
        'cell_length': cell_length,
        'sphere_radius': sphere_radius,
        'rho': rho
    }

    return Params_data

def read_Mkt_file(filepath):
    kVec = []
    times = []
    Mkt = []

    # read data from file
    with open(filepath, 'r') as txt_file:
        # ignore header line
        next(txt_file)

        # split data with comma separator
        line = txt_file.readline().split(',')

        k_points = int(line[0])
        time_samples = int(line[1])

        kVec = np.zeros([3, k_points])
        times = np.zeros(time_samples)
        Mkt = np.zeros([k_points, time_samples])

        # ignore next 2 lines
        next(txt_file)
        next(txt_file)
        for idx in range(k_points):
            line = txt_file.readline().split(',')
            kVec[0, idx] = float(line[1])
            kVec[1, idx] = float(line[2])
            kVec[2, idx] = float(line[3])
        
                # ignore next 2 lines
        next(txt_file)
        next(txt_file)
        for idx in range(time_samples):
            line = txt_file.readline().split(',')
            times[idx] = float(line[1])
        
                # ignore next 2 lines
        next(txt_file)
        next(txt_file)
        for kIdx in range(k_points):
            line = txt_file.readline().split(',')
            for tIdx in range(time_samples):
                Mkt[kIdx, tIdx] = float(line[tIdx + 1]) 
    
    Mkt_data = {
        'k': kVec,
        't': times,
        'Mkt': Mkt
    }

    return Mkt_data

def read_Dt_file(filepath):
    times = []
    Dt = []

    # read data from file
    with open(filepath, 'r') as txt_file:
        # ignore header line
        next(txt_file)

        # read and split next line
        while(line := txt_file.readline()):
            line = line.split('\n')
            line = line[0].split(',')
            times.append(float(line[1]))
            Dt.append(float(line[2]))
    
    Dt_data = {
        'times': np.array(times),
        'Dt': np.array(Dt)
    }

    return Dt_data

def plot_Mkt(Mkt_data, Params_data):
    diag = 1.0
    k_points = Mkt_data['k'].shape[1]
    time_samples = Mkt_data['t'].shape[0]

    ka = np.zeros(k_points) 
    for idx in range(k_points):
        ka[idx] = Params_data['cell_length'] * np.linalg.norm(Mkt_data['k'][:, idx]) 
        
    fig = plt.figure(figsize=(8,9), dpi=100)
    for time in range(time_samples):
        plt.semilogy(ka, Mkt_data['Mkt'][:, time], '-o', label=str(Mkt_data['t'][time]) + " ms")        
    plt.axvline(x=diag*np.pi, color="black", linewidth=0.5)
    plt.axvline(x=diag*3*np.pi, color="black", linewidth=0.5)
    plt.axvline(x=diag*5*np.pi, color="black", linewidth=0.5)
        
    plt.legend(loc='upper right')
    plt.xlabel(r'$ |k|a $')
    plt.ylabel(r'$ M(k,t) $')

    # Set plot axes limits
    plt.xlim(ka[0], 1.25*ka[-1])
    # plt.ylim(1.0e-07, 1.0)
    plt.show()
    return
    
def plot_Dt(Dt_data, Params_data):
    plt.plot(Dt_data['times'], Dt_data['Dt'], 'o', color='navy')
    plt.xlabel('time (msec)')
    plt.ylabel(r'$D(t)/D_{0}$')
    plt.ylim([0,1])
    plt.show()

    tAdim = (Params_data['Dp']/Params_data['cell_length']**2) * Dt_data['times']
    plt.plot(np.sqrt(tAdim), Dt_data['Dt'], 'o', color='darkred')
    plt.xlabel('$[D_{0}t/a^{2}]^{1/2}$')
    plt.ylabel(r'$D(t)/D_{0}$')
    plt.ylim([0,1])
    plt.show()
    return

def Main():
    # Read input parameters data from .txt file
    Params_filepath = r'./db/temp/Parameters.txt'
    Params_data = read_parameters_file(Params_filepath)

    # Read and plot M(k,t) data from .txt file
    Mkt_filepath = r'./db/temp/Mkt.txt'
    Mkt_data = read_Mkt_file(Mkt_filepath)
    plot_Mkt(Mkt_data, Params_data)

    # Read and plot D(t) data from .txt file
    Dt_filepath = r'./db/temp/Dt.txt'
    Dt_data = read_Dt_file(Dt_filepath)
    plot_Dt(Dt_data, Params_data) 
    
    return

if __name__ == '__main__':
    Main()
