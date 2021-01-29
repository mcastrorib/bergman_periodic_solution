import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
from LeastSquaresRegression import LeastSquaresRegression

def dataviz_Mkt(Mkt, k, a, labels, diag=1.0 ,title=''):
        # Points (s=point_size, c=color, cmap=colormap)
        times = len(Mkt)
        data_size = len(Mkt[0])
       
        ka = [] 
        for idx in range(data_size):
            ka.append(np.linalg.norm(k[idx]) * a) 
         
        fig = plt.figure(figsize=(8,9), dpi=100)
        for idx in range(times):
            plt.semilogy(ka, Mkt[idx], '-o', label=labels[idx] + " ms")        
        plt.axvline(x=diag*np.pi, color="black", linewidth=0.5)
        plt.axvline(x=diag*3*np.pi, color="black", linewidth=0.5)
        plt.axvline(x=diag*5*np.pi, color="black", linewidth=0.5)
            
        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlabel(r'$ |k|a $')
        plt.ylabel(r'$ M(k,t) $')

        # Set plot axes limits
        plt.xlim(ka[0], 1.25*ka[-1])
        # plt.ylim(1.0e-07, 1.0)
        plt.show()
        return

def dataviz_Dt(Dt, time, title=''):
        # Points (s=point_size, c=color, cmap=colormap)
         
        fig = plt.figure(figsize=(8,9), dpi=100)
        plt.plot(1e-3*time, Dt)        
        # plt.axvline(x=np.pi, color="black", linewidth=0.5)
            
        plt.title(title)
        plt.xlabel(r'$ time (sec) $')
        plt.ylabel(r'$ D(t)/D_{p} $')

        # Set plot axes limits
        plt.xlim(0.0, 1e-3*time[-1])
        plt.ylim(0.5,0.8)
        plt.show()
        return

def dataviz_vals_and_weights(vals, weights, k_array, a, nvals):       
    k_points = k_array.shape[0]
    vec_ka = np.array([a*np.linalg.norm(k_array[i]) for i in range(k_points)])
    
    fig, axs = plt.subplots(1, 2)
    for i in range(nvals):
        axs[0].plot(vec_ka, (a**2/D_p)*vals[i,:],'o', linewidth=0.5, label=str(i)+"q")

    axs[0].axvline(x=np.pi, color="black", linewidth=0.5)
    axs[0].axvline(x=3*np.pi, color="black", linewidth=0.5)
    axs[0].axvline(x=5*np.pi, color="black", linewidth=0.5)
    axs[0].legend(loc='upper right')
    axs[0].set_xlim([0,(1.25 * vec_ka[-1])])

    for i in range(nvals):
        axs[1].plot(vec_ka, np.abs(weights[i,:])**2,'-o', linewidth=0.5, label=str(i)+"q")

    axs[1].axvline(x=np.pi, color="black", linewidth=0.5)
    axs[1].axvline(x=3*np.pi, color="black", linewidth=0.5)
    axs[1].axvline(x=5*np.pi, color="black", linewidth=0.5)
    axs[1].legend(loc='upper right')
    axs[1].set_xlim([0,(1.25 * vec_ka[-1])])
    plt.show()
    return

def dataviz_vals_histogram(vals, weights, spur, k_point, times, a, Dp, porosity, path=None):
    fig, axs = plt.subplots(4, 1, figsize=(16,16))
    fig.suptitle("k index = " + str(k_point))
    axs[0].semilogy((a**2/Dp) * np.real(vals[:, k_point]), 'o')
    axs[0].set_title("Reduced eigenvalues ($\lambda$)")
    axs[0].set_xlim([-1,vals.shape[0]])
    # axs[3].set_ylim([0.00001,10000])
    
    axs[1].semilogy(np.real(weights[:, k_point]),'o')
    axs[1].set_title("Weights ($\phi (g)$)")
    axs[1].set_xlim([-1,vals.shape[0]])

    product = np.zeros(vals.shape[0], dtype=complex)
    for time in times:
        for i in range(vals.shape[0]):
            product[i] = (1.0/porosity) * np.real(np.exp((-1.0)* vals[i, k_point] * time) * np.abs(weights[i, k_point])**2)
        axs[2].semilogy(np.real(product),'o', label=str(time) + "ms")

    axs[2].set_title(r'$e^{-\lambda t}|\phi(g)|$')
    axs[2].set_xlim([-1,vals.shape[0]])
    axs[2].set_ylim([0.01*np.real(product).max(), 1.e+0])
    axs[2].legend(loc='upper right')

    axs[3].plot(np.real(spur[:, k_point]),'o')
    axs[3].set_title("Spurious quantity")
    axs[3].set_xlim([-1,vals.shape[0]])
    axs[3].set_ylim([0,1])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    if(path == None):
        plt.show()
    else:
        complete_path = path + 'k_point=' + str(k_point) + '.png'
        plt.savefig(complete_path)
        plt.show()
    return     

def plot_fft_3d_results(signal, dft, fft, nimgs=1):  
    diff_abs = np.abs(dft)-np.abs(fft)
    diff_real = np.real(dft)-np.real(fft)
    diff_imag = np.imag(dft)-np.imag(fft)
    cmap = cm.PRGn
    cmap=cm.get_cmap(cmap)

    points = signal.shape[0]
    fig, axs = plt.subplots(nimgs, 10)
    img_list = []
    for im in range(nimgs):
        im00 = axs[im,0].imshow(np.abs(signal[im]), cmap=cmap)
        im01 = axs[im,1].imshow(np.abs(dft[im]), cmap=cmap)
        im02 = axs[im,2].imshow(np.abs(fft[im]), cmap=cmap)
        im03 = axs[im,3].imshow(np.real(dft[im]), cmap=cmap)
        im04 = axs[im,4].imshow(np.real(fft[im]), cmap=cmap)
        im05 = axs[im,5].imshow(np.imag(dft[im]), cmap=cmap)
        im06 = axs[im,6].imshow(np.imag(fft[im]), cmap=cmap)
        im07 = axs[im,7].imshow(diff_real[im], cmap=cmap)
        im08 = axs[im,8].imshow(diff_imag[im], cmap=cmap)
        im09 = axs[im,9].imshow(diff_abs[im], cmap=cmap)             

        fig.colorbar(im00, ax=axs[im,0])
        fig.colorbar(im01, ax=axs[im,1])
        fig.colorbar(im02, ax=axs[im,2])
        fig.colorbar(im03, ax=axs[im,3])
        fig.colorbar(im04, ax=axs[im,4])
        fig.colorbar(im05, ax=axs[im,5])
        fig.colorbar(im06, ax=axs[im,6])
        fig.colorbar(im07, ax=axs[im,7])
        fig.colorbar(im08, ax=axs[im,8])
        fig.colorbar(im09, ax=axs[im,9])
    
    axs[0,0].set_title("signal")
    axs[0,1].set_title("abs(dft)")
    axs[0,2].set_title("abs(fft)")
    axs[0,3].set_title("re(dft)")
    axs[0,4].set_title("re(fft)")
    axs[0,5].set_title("im(dft)")
    axs[0,6].set_title("im(fft)")
    axs[0,7].set_title("re(diff)")
    axs[0,8].set_title("im(diff)")
    axs[0,9].set_title("abs(diff)")
    
    for im in range(nimgs):
        for col in range(6):
            axs[im,col].grid(False)

            # Hide axes ticks
            axs[im,col].set_xticks([])
            axs[im,col].set_yticks([])
    
    # fig.tight_layout()
    plt.show()
    return

def normalize_signal_3d(signal):
    nsignal = np.zeros([signal.shape[0], signal.shape[1], signal.shape[2]], dtype=complex)
    s0 = signal.max()

    for k in range(signal.shape[2]):
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                nsignal[i,j,k] = signal[i,j,k]/s0
    return nsignal

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

# Functions
def pore_function(x,y,z,R):
    point = np.array([x,y,z])
    if(np.linalg.norm(point) > R):
        return 1.0
    else:
        return 0.0

def matrix_function(x,y,z,R):
    point = np.array([x,y,z])
    if(np.linalg.norm(point) < R):
        return 1.0
    else:
        return 0.0

def set_porosity(a, R):
    return 1.0 + (np.pi / 4.0) - (3.0 * np.pi * (R/a)**2) + ((8.0 / 3.0 ) * np.pi) * (R/a)**3 

def apply_fft(signal):
    kspec = np.fft.fftn(signal, norm='ortho')
    kspec = np.fft.fftshift(kspec)
    return kspec

def apply_dft(signal, dr, dg, volume, points):
    kspec = np.zeros([points, points, points], dtype=complex)
    elems = points**3
    dV = volume / (elems)
    count = 0    
    for k in range(points):
        for i in range(points):
            for j in range(points):
                count += 1
                print(":: Pore_dg {} out of {}.".format(count, elems))
                dG = dg[i,j,k]
                gsum = 0.0
                for rz in range(points):
                    for rx in range(points):
                        for ry in range(points):
                            gsum += dV * signal[rx, ry, rz] * np.exp((-1.0j) * np.dot(dG, dr[rx,ry,rz]))
                
                kspec[i,j,k] = (1.0 / volume) * gsum
    
    return kspec

def apply_identity(ksignal, dpoints):
    new_signal = np.zeros([dpoints, dpoints, dpoints], dtype=complex)
    count = 0
    elems = dpoints**3
    for k in range(dpoints):
        for j in range(dpoints):
            for i in range(dpoints):
                print(":: Matrix_dg {} out of {}.".format(count, elems))
                new_signal[i,j,k] = (-1.0) * ksignal[i,j,k]
                count += 1
    
    new_signal[dpoints//2, dpoints//2, dpoints//2] += 1.0
    return new_signal

def IDX2C_3D(i, j, k, dim):
    return ((k * (dim * dim)) + ((i) * (dim)) + (j))

def map_g_to_dg(idx1, idx2, N):
    # new_i1 = idx1 - N
    # new_i2 = idx2 - N
    # final_idx = new_i1 - new_i2
    # final_idx += 2*N
    # return final_idx
    return (idx1 - idx2 + 2*N)

def find_gk(kvec, g, points):
    half = points//2
    gk_index = IDX2C_3D(half, half, half, points)
    gkvec = g[half, half, half]
    diffvec = kvec - gkvec
    distance = np.sqrt(np.dot(diffvec, diffvec))

    kxmin = half
    kxmax = points
    kxinc = 1
    if(kvec[0] < 0.0):
        kxmax = -1
        kxinc = -1
    
    kymin = half
    kymax = points
    kyinc = 1
    if(kvec[1] < 0.0):
        kymax = -1
        kyinc = -1
    
    kzmin = half
    kzmax = points
    kzinc = 1
    if(kvec[2] < 0.0):
        kzmax = -1
        kzinc = -1
    
    # print("k = ", kvec)
    # print(gk_index, gkvec, distance)
    # print(kxmin, kxmax, kxinc)
    # print(kymin, kymax, kyinc)
    # print(kzmin, kzmax, kzinc)
    for k in range(kzmin, kzmax, kzinc):
        for i in range(kymin, kymax, kyinc):
            for j in range(kxmin, kxmax, kxinc):
                
                diffvec = kvec - g[i, j, k]
                new_distance = np.sqrt(np.dot(diffvec, diffvec))
                # print("i,j,k = ", i,j,k, "G = ", g[j, i, k], new_distance)

                if(new_distance < distance):
                    distance = new_distance
                    gkvec = g[i, j, k]
                    gk_index = IDX2C_3D(i, j, k, points)
                    # print(gk_index, gkvec, distance)             
    
    return gkvec, gk_index
    
def Dt_recover(Mkt, k, times, Dp, points=10):
    time_samples = len(times)
    Dts = np.zeros(time_samples)
    ksquare = np.array([np.linalg.norm(k_array[i])**2 for i in range(points)])
    # print(ksquare)
    for t in range(time_samples):
        logMkt = (-1.0) * np.log(Mkt[t, 0:points])
        Dpk2t = Dp * times[t] * ksquare
        # print(Dpk2t)
        lsa = LeastSquaresRegression()
        lsa.config(Dpk2t, logMkt, points)
        lsa.solve()
        
        Dts[t] = lsa.get_B()
        # Dts[t] = ((-1.0) * np.log(Mkt[t, -1])) / (Dp * times[t] * np.linalg.norm(k)**2)
    return Dts

def Dt_recover_paper(Mkt, k, times, Dp):
    time_samples = len(times)
    Dts = np.zeros(time_samples)
    for t in range(time_samples):
        Dts[t] = (-1.0) * np.log(Mkt[t, 0]) / (Dp * np.linalg.norm(k[0])**2 * times[t])
    return Dts

def order_vals_weights_spurs(vals, weights, spurs):
    size = vals.shape[0]
    k_points = vals.shape[1]

    low_vals = np.zeros([size, k_points], dtype=complex)
    low_weights = np.zeros([size, k_points], dtype=complex)
    low_spur = np.zeros([size, k_points], dtype=complex)
    for k in range(k_points):
        indexes = np.argsort(vals[:,k])
        for val in range(size):
            low_vals[val, k] = vals[indexes[val], k]
            low_weights[val, k] = weights[indexes[val], k]
            low_spur[val, k] = spurs[indexes[val], k]
    return low_vals, low_weights, low_spur

def get_true_vals(vals, weights, spurs, spurious_cut, nvals):    
    size = vals.shape[0]
    k_points = vals.shape[1]
    low_vals, low_weights, low_spur = order_vals_weights_spurs(vals, weights, spurs)

    true_low_vals = np.zeros([nvals, k_points], dtype=complex)
    true_low_weights = np.zeros([nvals, k_points], dtype=complex)
    true_low_spurs = np.zeros([nvals, k_points], dtype=complex)
    for k in range(k_points):
        vals_added = 0
        row = 0
        spur_count = 0
        
        while(vals_added < nvals and row < size):
            if(np.real(low_spur[row, k]) > spurious_cut):
                true_low_vals[vals_added, k] = low_vals[row, k]
                true_low_weights[vals_added, k] = low_weights[row, k]
                true_low_spurs[vals_added, k] = low_spur[row, k]
                vals_added += 1
            else:
                spur_count += 1
            row += 1
        
        print("k = {}, {} spurious eigenvalues".format(k, spur_count))

        if(vals_added < nvals):
            print("not enough values :/")
    return true_low_vals, true_low_weights, true_low_spurs


# Problem parameters
D_p = 2.5   # in um²/ms
D_m = 0.0   # in um²/ms
a = 10.0    # in um
Radius = 5.0     # in um
N = int(3)
w = 0.9999
u = 1.0
spurious_cut = 0.2

k_direction = np.array([1,0,0])
# ka_min = 1.0
# ka_max = 5*np.pi
ka_min = 0.5
ka_max = 1.0
k_points = 2
k_array = np.zeros([k_points, 3])
k_linspace = np.linspace(ka_min/a, ka_max/a, k_points)
# k_linspace = np.flip(k_linspace)
for i in range(3):
    k_array[:, i] = k_direction[i] * k_linspace
# k_array = np.flip(k_array)


time_samples = 50
times = np.logspace(-1,2,time_samples) # in ms
# times = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0, 100.0])
Mkt = np.zeros([time_samples, k_points]) 

volume = a**3
porosity = set_porosity(a, Radius) 
points = 2*N + 1
Nrange = np.arange(-N, N+1)

rfreq = 0.5*a / float(N)
vecX = np.linspace(-0.5*a, 0.5*a, 2*points + 1)[1:2*points:2]
vecY = np.linspace(-0.5*a, 0.5*a, 2*points + 1)[1:2*points:2]
vecZ = np.linspace(-0.5*a, 0.5*a, 2*points + 1)[1:2*points:2]


gFreq = 2*np.pi/a
[vecGX] = np.meshgrid(gFreq*np.arange(-N, N+1))
[vecGY] = np.meshgrid(gFreq*np.arange(-N, N+1))
[vecGZ] = np.meshgrid(gFreq*np.arange(-N, N+1))

poreR = np.array(np.zeros([points, points, points]))
matrixR = np.array(np.zeros([points, points, points]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            pX = vecX[j]
            pY = vecY[i]
            pZ = vecZ[k]
            poreR[i,j,k] = pore_function(pX, pY, pZ, Radius)
            matrixR[i,j,k] = matrix_function(pX, pY, pZ, Radius)

gridR = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            gridR[i, j, k, 0] = vecY[i]
            gridR[i, j, k, 1] = vecX[j] 
            gridR[i, j, k, 2] = vecZ[k] 

gridG = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            gridG[i, j, k, 0] = vecGY[i] 
            gridG[i, j, k, 1] = vecGX[j]
            gridG[i, j, k, 2] = vecGZ[k] 

# g - g' space
dpoints = 4*N + 1            
vecDX = np.linspace(-0.5*a, 0.5*a, 2*dpoints + 1)[1:2*dpoints:2]
vecDY = np.linspace(-0.5*a, 0.5*a, 2*dpoints + 1)[1:2*dpoints:2]
vecDZ = np.linspace(-0.5*a, 0.5*a, 2*dpoints + 1)[1:2*dpoints:2]
[vecDGX] = np.meshgrid(gFreq*np.arange(-2*N, 2*N+1))
[vecDGY] = np.meshgrid(gFreq*np.arange(-2*N, 2*N+1))
[vecDGZ] = np.meshgrid(gFreq*np.arange(-2*N, 2*N+1))

gridDR = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            gridDR[i, j, k, 0] = vecDY[i] 
            gridDR[i, j, k, 1] = vecDX[j]
            gridDR[i, j, k, 2] = vecDZ[k]

gridDG = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            gridDG[i, j, k, 0] = vecDGY[i] 
            gridDG[i, j, k, 1] = vecDGX[j]
            gridDG[i, j, k, 2] = vecDGZ[k] 

poreDR = np.array(np.zeros([dpoints, dpoints, dpoints]))
matrixDR = np.array(np.zeros([dpoints, dpoints, dpoints]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            pX = vecDX[j]
            pY = vecDY[i]
            pZ = vecDZ[k]
            poreDR[i,j,k] = pore_function(pX, pY, pZ, Radius)
            matrixDR[i,j,k] = matrix_function(pX, pY, pZ, Radius)

# -- Apply Fourier Transform
# Apply FFT-3D to characteristic signal
# pore_dg = apply_fft(pore_dr)
# pore_dg = (1.0/volume) * pore_dg
# pore_dg = normalize_signal_3d(pore_dg)
# matrix_dg = apply_fft(matrix_dr)
# matrix_dg = (1.0/volume) * matrix_dg
# matrix_dg = normalize_signal_3d(matrix_dg)

poreDG = apply_dft(poreDR, gridDR, gridDG, volume, dpoints)
# pore_dg = normalize_signal_3d(pore_dg)
# matrixDG = apply_dft(matrixDR, gridDR, gridDG, volume, dpoints)
matrixDG = apply_identity(poreDG, dpoints)
# matrix_dg = normalize_signal_3d(matrix_dg)

# -- Matrices Assembly
rows = points**3
cols = rows
matW = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matU = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matT = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matR = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matRinv = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matRH = np.asmatrix(np.zeros([rows, cols], dtype=complex))
matRHinv = np.asmatrix(np.zeros([rows, cols], dtype=complex))

# W matrix
occurs = np.zeros([dpoints, dpoints, dpoints])
for k in range(points):
    for i in range(points):
        for j in range(points):
            row_index = IDX2C_3D(i, j, k, points)
            print(":: Matrix_W row {} out of {}.".format(row_index, rows))
            
            for kk in range(points):
                for ii in range(points):
                    for jj in range(points):
                        col_index = IDX2C_3D(ii, jj, kk, points)
                        # print(":: :: col {} out of {}.".format(col_index, cols))

                        di = map_g_to_dg(i, ii, N)
                        dj = map_g_to_dg(j, jj, N)
                        dk = map_g_to_dg(k, kk, N)
                        occurs[di, dj, dk] += 1
                        matW[row_index, col_index] = (-1.0) * w * matrixDG[di, dj, dk]

for row in range(rows):
    matW[row, row] += 1.0

# Get cholesky decomposition R matrix
matRH = np.linalg.cholesky(matW)
matRHinv = np.linalg.inv(matRH)
matR = matRH.H
matRinv = np.linalg.inv(matR)
matRRH = (matR * matRH)

# U matrix
for k in range(points):
    for i in range(points):
        for j in range(points):
            row_index = IDX2C_3D(i, j, k, points)
            
            for kk in range(points):
                for ii in range(points):
                    for jj in range(points):
                        col_index = IDX2C_3D(ii, jj, kk, points)
                        di = map_g_to_dg(i,ii,N)
                        dj = map_g_to_dg(j,jj,N)
                        dk = map_g_to_dg(k,kk,N)
                        matU[row_index, col_index] = (-1.0) * u * matrixDG[di, dj, dk]

for row in range(rows):
    matU[row, row] += 1.0
                        
# Solve for q
vals_q = np.zeros([rows, k_points], dtype=complex)
weights_q = np.zeros([rows, k_points], dtype=complex)
spur_q = np.zeros([rows, k_points], dtype=complex)   

for k_index in range(k_points):
    kvec = k_array[k_index]

    # Find gk in reciprocal lattice
    gkvec, gk_index = find_gk(kvec, gridG, points)
    qvec = kvec - gkvec


    print("-------------------------")
    print("k_index = \t", k_index)
    print("k = \t", kvec)
    print("gk = \t", gkvec)
    print("gk_index = \t", gk_index)
    print("q = \t", qvec)
    print("-------------------------")

    # T matrix
    for k in range(points):
        for i in range(points):
            for j in range(points):
                row_index = IDX2C_3D(i, j, k, points)
                qgRow = qvec + gridG[i,j,k]
                
                for kk in range(points):
                    for ii in range(points):
                        for jj in range(points):
                            col_index = IDX2C_3D(ii, jj, kk, points)
                            qgCol = qvec + gridG[ii,jj,kk]
                            matT[row_index, col_index] = np.dot(qgRow, qgCol) * matU[row_index, col_index] 

    # V matrix
    matV = D_p * (matRinv.H * matT * matRinv)
    if(check_symmetric(matV)):
        print("matV is symmetric")
    else:
        print("matV is not symmetric")
    vals, vecs = np.linalg.eigh(matV)
    # vec = vecs.transpose()
    # Sort eigen values and its vector
    # inds = np.argsort(vals)
    # vals = vals[inds]
    # vecs = vecs[inds] 
    weights = np.zeros([rows, cols])
    matAux = (1.0/w) * (matRH - ((1-w) * matRinv))
    weights =  matAux * vecs

    # persistent data
    for row in range(rows):
        vals_q[row, k_index] = vals[row]
        weights_q[row, k_index] = weights[gk_index, row]

    for n in range(rows):
        value = ((vecs[:,n].H * matRRH) * vecs[:,n])
        spur_q[n, k_index] = value[0,0]      

    # M(k,t)
    for t_idx in range(time_samples):
        Mkt_sum = 0.0
        for n in range(points**3):
            # if(spur_q[n, k_index] > spurious_cut):
            Mkt_sum += np.exp((-1.0) * vals[n] * times[t_idx]) * (np.abs(weights[gk_index, n]))**2
        Mkt[t_idx, k_index] = (1.0 / porosity) * np.real(Mkt_sum)


    # reduced_vals = (a**2 / D_p) * vals
    # plt.semilogy(np.real(quantity), np.real(reduced_vals), 'o')
    # plt.xlim([0,1])
    # plt.xlabel(r'$ (W \phi) ^{2}  $')
    # plt.ylabel(r'$ Reduced \, eigenvalues $')
    # plt.show()

# Normalize M(k,t):
normMkt = np.zeros([time_samples, k_points])
for time in range(time_samples):
    M0t = Mkt[time, 0]
    for k in range(k_points):
        normMkt[time, k] = Mkt[time,k] / M0t

# Data visualization
time_labels = np.array([str(time) for time in times])
dataviz_Mkt(Mkt, k_array, a, time_labels, np.sqrt(1.0))

values = 30
low_vals, low_weights, low_spurs = order_vals_weights_spurs(vals_q, weights_q, spur_q)
true_vals, true_weights, true_spur = get_true_vals(vals_q, weights_q, spur_q, spurious_cut, values)
dataviz_vals_and_weights(np.real(true_vals), np.real(true_weights), k_array, a, values)
dataviz_vals_histogram(low_vals, low_weights, low_spurs, 0, times[0:time_samples:2], a, D_p, porosity)
dataviz_vals_histogram(low_vals, low_weights, low_spurs, 1, times[0:time_samples:2], a, D_p, porosity)

Dt_points = k_points
Dts = Dt_recover(Mkt, k_array, times, D_p, Dt_points)
dataviz_Dt(Dts, times)
Dts5 = Dt_recover(normMkt, k_array, times, D_p, 2)
dataviz_Dt(Dts5, times)

new_Mkt = np.zeros([time_samples, k_points])
for k_idx in range(k_points):
    for t_idx in range(time_samples):
        Mkt_sum = 0.0
        for n in range(points**3):
            # if(spur_q[n, k_index] > spurious_cut):
            Mkt_sum += np.exp((-1.0) * low_vals[n, k_idx] * times[t_idx]) * (np.abs(low_weights[n, k_idx]))**2
        new_Mkt[t_idx, k_idx] = (1.0 / porosity) * np.real(Mkt_sum)


dataviz_Mkt(new_Mkt, k_array, a, time_labels, np.sqrt(1.0))

new_true_Mkt = np.zeros([time_samples, k_points])
for k_idx in range(k_points):
    for t_idx in range(time_samples):
        Mkt_sum = 0.0
        for n in range(values):
            # if(spur_q[n, k_index] > spurious_cut):
            Mkt_sum += np.exp((-1.0) * true_vals[n, k_idx] * times[t_idx]) * (np.abs(true_weights[n, k_idx]))**2
        new_true_Mkt[t_idx, k_idx] = (1.0 / porosity) * np.real(Mkt_sum)


dataviz_Mkt(new_true_Mkt, k_array, a, time_labels, np.sqrt(1.0))

reduced_vals = (1.0 / D_p) * np.real(true_vals)
q2 = k_array[:,0] * k_array[:,0] 

B = (1.0 / (q2[1] - q2[0])) * (reduced_vals[:, 1] - reduced_vals[:, 0])
A = reduced_vals[:,0] - (B * q2[0])
A = a**2 * A