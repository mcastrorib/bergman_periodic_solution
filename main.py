import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
        plt.ylim(0.5,1.2)
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
        axs[1].plot(vec_ka, weights[i,:],'-o', linewidth=0.5, label=str(i)+"q")

    axs[1].axvline(x=np.pi, color="black", linewidth=0.5)
    axs[1].axvline(x=3*np.pi, color="black", linewidth=0.5)
    axs[1].axvline(x=5*np.pi, color="black", linewidth=0.5)
    axs[1].legend(loc='upper right')
    axs[1].set_xlim([0,(1.25 * vec_ka[-1])])
    plt.show()
    return

def dataviz_vals_histogram(vals, weights, spur, k_point, times, a, Dp, porosity):
    fig, axs = plt.subplots(4, 1)
    fig.suptitle("k index = " + str(k_point))
    axs[0].semilogy((a**2/Dp) * np.real(vals[:, k_point]), 'o')
    axs[0].set_title("Reduced eigenvalues ($\lambda$)")
    axs[0].set_xlim([-1,vals.shape[0]])
    # axs[3].set_ylim([0.00001,10000])
    
    axs[1].semilogy(np.real(weights[:, k_point]),'o')
    axs[1].set_title("Weights ($\phi (g)$)")
    axs[1].set_xlim([-1,vals.shape[0]])

    product = np.zeros(vals.shape[0])
    for time in times:
        for i in range(vals.shape[0]):
            product[i] = (1.0/porosity) * np.real(np.exp((-1.0)*vals[i, k_point]*time) * weights[i, k_point])
        axs[2].semilogy(product,'o', label=str(time) + "ms")
    axs[2].set_title(r'$e^{-\lambda t}|\phi(g)|$')
    axs[2].set_xlim([-1,vals.shape[0]])
    axs[2].set_ylim([1.e-6, 1.e+0])
    axs[2].legend(loc='upper right')

    axs[3].plot(np.real(spur[:, k_point]),'o')
    axs[3].set_title("Spurious quantity")
    axs[3].set_xlim([-1,vals.shape[0]])
    axs[3].set_ylim([0,1])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    plt.show()
    return     

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
    return ((k * (dim * dim)) + ((j) * (dim)) + (i))

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
        for j in range(kymin, kymax, kyinc):
            for i in range(kxmin, kxmax, kxinc):
                
                diffvec = kvec - g[j, i, k]
                new_distance = np.sqrt(np.dot(diffvec, diffvec))
                # print("i,j,k = ", i,j,k, "G = ", g[j, i, k], new_distance)

                if(new_distance < distance):
                    distance = new_distance
                    gkvec = g[j, i, k]
                    gk_index = IDX2C_3D(j, i, k, points)
                    # print(gk_index, gkvec, distance)             
    
    return gkvec, gk_index
    
def Dt_recover(Mkt, k, times, Dp, points=10):
    time_samples = len(times)
    Dts = np.zeros(time_samples)
    ksquare = np.array([np.linalg.norm(k_array[i])**2 for i in range(points)])
    print(ksquare)
    for t in range(time_samples):
        lsa = LeastSquaresRegression()
        logMkt = (-1.0) * np.log(Mkt[t, 0:points])
        Dpk2t = Dp * times[t] * ksquare
        lsa.config(Dpk2t, logMkt, points)
        lsa.solve()
        Dts[t] = lsa.get_B()
        # Dts[t] = ((-1.0) * np.log(Mkt[t, -1])) / (Dp * times[t] * np.linalg.norm(k)**2)
    return Dts

def order_vals_weights_spurs(vals, weights, spurs):
    size = vals.shape[0]
    k_points = vals.shape[1]

    low_vals = np.zeros([size, k_points], dtype=complex)
    low_weights = np.zeros([size, k_points])
    low_spur = np.zeros([size, k_points])
    for k in range(k_points):
        indexes = np.argsort(vals[:,k])
        for val in range(size):
            low_vals[val, k] = vals[indexes[val], k]
            low_weights[val, k] = np.abs(weights[indexes[val], k])**2
            low_spur[val, k] = np.real(spurs[indexes[val], k])
    return low_vals, low_weights, low_spur

def get_true_vals(vals, weights, spurs, spurious_cut, nvals):    
    size = vals.shape[0]
    k_points = vals.shape[1]
    low_vals, low_weights, low_spur = order_vals_weights_spurs(vals, weights, spurs)

    true_low_vals = np.zeros([nvals, k_points])
    true_low_weights = np.zeros([nvals, k_points])
    true_low_spurs = np.zeros([nvals, k_points])
    for k in range(k_points):
        vals_added = 0
        row = 0
        spur_count = 0
        
        while(vals_added < nvals and row < size):
            if(low_spur[row, k] > spurious_cut):
                true_low_vals[vals_added, k] = np.real(low_vals[row, k])
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
R = 5.0     # in um
N = int(3)
w = 0.9999
u = 1.0
spurious_cut = 0.25

k_direction = np.array([1,0,0])
ka_max = 2*np.pi
# ka_max = 0.1
k_points = 20
k_array = np.zeros([k_points, 3])
k_linspace = np.linspace(ka_max/a, 0.001, k_points)
k_linspace = np.flip(k_linspace)
for i in range(3):
    k_array[:, i] = k_direction[i] * k_linspace
# k_array = np.flip(k_array)


time_samples = 40
times = np.logspace(-1,2,time_samples) # in ms
# times = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0, 100.0]
Mkt = np.zeros([time_samples, k_points]) 

volume = a**3
porosity = set_porosity(a,R) 
points = 2*N + 1
Nrange = np.arange(-N, N+1)

rfreq = 0.5*a / float(N)
# [x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 
Rrange = np.linspace(-0.5*a, 0.5*a, 2*points + 1)[1:2*points:2]
[x, y, z] = np.meshgrid(Rrange, Rrange, Rrange)


gfreq = 2*np.pi/a
[gx, gy, gz] = np.meshgrid(gfreq*Nrange, gfreq*Nrange, gfreq*Nrange)

pore_r = np.array(np.zeros([points, points, points]))
matrix_r = np.array(np.zeros([points, points, points]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            px = x[i,j,k]
            py = y[i,j,k]
            pz = z[i,j,k]
            pore_r[i,j,k] = pore_function(px, py, pz, R)
            matrix_r[i,j,k] = matrix_function(px, py, pz, R)

r = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            r[i, j, k, 0] = x[i, j, k] 
            r[i, j, k, 1] = y[i, j, k]
            r[i, j, k, 2] = z[i, j, k] 

g = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for i in range(points):
        for j in range(points):
            g[i, j, k, 0] = gx[i, j, k] 
            g[i, j, k, 1] = gy[i, j, k]
            g[i, j, k, 2] = gz[i, j, k] 

# g - g' space
dpoints = 4*N + 1            
dNrange = np.arange(-2*N, 2*N+1)
dRrange = np.linspace(-0.5*a, 0.5*a, 2*dpoints + 1)[1:2*dpoints:2]
drfreq = 0.5*rfreq
norm_factor = 1.0
while (norm_factor < dpoints):
    norm_factor *= 2.0

[dg_x, dg_y, dg_z] = np.meshgrid(gfreq * dNrange, gfreq * dNrange, gfreq * dNrange)
[dx, dy, dz] = np.meshgrid(dRrange, dRrange, dRrange)
# [dx, dy, dz] = np.meshgrid(drfreq * dNrange, drfreq * dNrange, drfreq * dNrange) 

dr = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            dr[i, j, k, 0] = dx[i, j, k] 
            dr[i, j, k, 1] = dy[i, j, k]
            dr[i, j, k, 2] = dz[i, j, k] 

dg = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            dg[i, j, k, 0] = dg_x[i, j, k] 
            dg[i, j, k, 1] = dg_y[i, j, k]
            dg[i, j, k, 2] = dg_z[i, j, k] 

pore_dr = np.array(np.zeros([dpoints, dpoints, dpoints]))
matrix_dr = np.array(np.zeros([dpoints, dpoints, dpoints]))
for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            px = dx[i,j,k]
            py = dy[i,j,k]
            pz = dz[i,j,k]
            pore_dr[i,j,k] = pore_function(px, py, pz, R)
            matrix_dr[i,j,k] = matrix_function(px, py, pz, R)

# Apply FFT-3D to characteristic signal
# pore_dg = apply_fft(pore_dr)
# pore_dg = (1.0/volume) * pore_dg
# matrix_dg = apply_fft(matrix_dr)
# matrix_dg = (1.0/volume) * matrix_dg

pore_dg = apply_dft(pore_dr, dr, dg, volume, dpoints)
# matrix_dg = apply_dft(matrix_dr, dr, dg, volume, dpoints)
matrix_dg = apply_identity(pore_dg, dpoints)

# Assembly matrices
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

                        di = map_g_to_dg(i,ii,N)
                        dj = map_g_to_dg(j,jj,N)
                        dk = map_g_to_dg(k,kk,N)
                        occurs[di, dj, dk] += 1
                        matW[row_index, col_index] = (-1.0) * w * matrix_dg[di, dj, dk]

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
                        matU[row_index, col_index] = (-1.0) * u * matrix_dg[di, dj, dk]

for row in range(rows):
    matU[row, row] += 1.0
                        
# Solve for q
# for q in q_array:
real_spurious = np.zeros([rows, k_points], dtype=complex)    
vals_q = np.zeros([rows, k_points], dtype=complex)
weights_q = np.zeros([rows, k_points], dtype=complex)
first_brillouin = (0.5) * g[points//2 + 1, points//2 + 1, points//2 + 1][0]
for k_index in range(k_points):
    kvec = k_array[k_index]

    # Find gk in reciprocal lattice
    gkvec, gk_index = find_gk(kvec, g, points)
    
    # Set q vec in first brillouin zone
    # qvec = np.zeros(3)
    # for i in range(3):
    #     value = kvec[i] - gkvec[i]
    #     if(value > first_brillouin):
    #         value = value % first_brillouin
    #     elif(value < (-1.0) * first_brillouin):
    #         value = np.abs(value % ((-1.0)*first_brillouin))
    #     qvec[i] = value
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
                qgRow = qvec + g[i,j,k]
                
                for kk in range(points):
                    for ii in range(points):
                        for jj in range(points):
                            col_index = IDX2C_3D(ii, jj, kk, points)
                            qgCol = qvec + g[ii,jj,kk]
                            matT[row_index, col_index] = np.dot(qgRow, qgCol) * matU[row_index, col_index] 

    # V matrix
    matV = D_p * (matRinv.H * matT * matRinv)
    vals, vecs = np.linalg.eig(matV)
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
        real_spurious[n, k_index] = value[0,0]      

    # M(k,t)
    for t_idx in range(time_samples):
        Mkt_sum = 0.0
        for n in range(points**3):
            # if(real_spurious[n, k_index] > spurious_cut):
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
# dataviz_Mkt(Mkt, k_array, a, time_labels, np.sqrt(1.0))

points = 4
Dts = Dt_recover(Mkt, k_array, times, D_p, points)
dataviz_Dt(Dts, times)

values = 50
low_vals, low_weights, low_spurs = order_vals_weights_spurs(vals_q, weights_q, real_spurious)
true_vals, true_weights, true_spur = get_true_vals(vals_q, weights_q, real_spurious, spurious_cut, values)
# dataviz_vals_and_weights(true_vals, true_weights, k_array, a, values)
dataviz_vals_histogram(low_vals, low_weights, low_spurs, 0, times[:time_samples:8], a, D_p, porosity)