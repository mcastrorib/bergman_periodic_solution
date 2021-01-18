import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Functions
def pore_function(x,y,z,R):
    point = np.array([x,y,z])
    if(np.linalg.norm(point) > R):
        return 1.0
    else:
        return 0.0

def matrix_function(x,y,z,R):
    point = np.array([x,y,z])
    if(np.linalg.norm(point) <= R):
        return 1.0
    else:
        return 0.0

def apply_fft(signal):
    ksignal = np.fft.fftn(signal)
    ksignal = np.fft.fftshift(ksignal)
    return ksignal

def apply_dft(signal, dr, dg, volume, dpoints):
    ksignal = np.zeros([dpoints, dpoints, dpoints], dtype=complex)
    dV = volume / (dpoints**3)
    count = 0
    elems = dpoints**3
    for k in range(dpoints):
        for j in range(dpoints):
            for i in range(dpoints):
                count += 1
                print(":: Pore_dg {} out of {}.".format(count, elems))
                dG = dg[i,j,k]
                gsum = 0.0
                for rz in range(dpoints):
                    for ry in range(dpoints):
                        for rx in range(dpoints):
                            gsum += dV * signal[rx, ry, rz] * np.exp((-1.0j) * np.dot(dG, dr[rx,ry,rz]))
                
                ksignal[i,j,k] = (1.0 / volume) * gsum
    
    return ksignal

def apply_identity(ksignal, dpoints):
    new_signal = np.zeros([dpoints, dpoints, dpoints], dtype=complex)
    count = 0
    elems = dpoints**3
    for k in range(dpoints):
        for j in range(dpoints):
            for i in range(dpoints):
                print(":: Matrix_dg {} out of {}.".format(count, elems))
                new_signal[i,j,k] = (-1.0) * ksignal[i,j,k]
    
    new_signal[dpoints//2, dpoints//2, dpoints//2] += 1.0
    return new_signal

def IDX2C_3D(i, j, k, dim):
    return ((k * (dim * dim)) + ((j) * (dim)) + (i))

def map_g_to_dg(idx1, idx2, N):
    new_i1 = idx1 - N
    new_i2 = idx2 - N
    final_idx = new_i1 - new_i2
    final_idx += 2*N
    return final_idx
    


# Problem parameters
D_p = 2.5
D_m = 0.0
a = 10.0
R = 5.0
N = int(1)
w = 0.9999
u = 1.0
q_array = np.array([[0.01, 0.0, 0.0]])

volume = a**3
porosity = 1.0 + (np.pi / 4.0) - (3.0 * np.pi * (R/a)**2) + ((8.0 / 3.0 ) * np.pi) * (R/a)**3 
points = 2*N + 1
Nrange = np.arange(-N, N+1)

rfreq = 0.5*a / float(N)
[x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 

gfreq = 2*np.pi/a
[gx, gy, gz] = np.meshgrid(gfreq*Nrange, gfreq*Nrange, gfreq*Nrange)

pore_r = np.array(np.zeros([points, points, points]))
matrix_r = np.array(np.zeros([points, points, points]))
for k in range(points):
    for j in range(points):
        for i in range(points):
            px = x[i,j,k]
            py = y[i,j,k]
            pz = z[i,j,k]
            pore_r[i,j,k] = pore_function(px, py, pz, R)

for k in range(points):
    for j in range(points):
        for i in range(points):
            px = x[i,j,k]
            py = y[i,j,k]
            pz = z[i,j,k]
            matrix_r[i,j,k] = matrix_function(px, py, pz, R)

r = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for j in range(points):
        for i in range(points):
            r[i, j, k, 0] = x[i, j, k] 
            r[i, j, k, 1] = y[i, j, k]
            r[i, j, k, 2] = z[i, j, k] 

g = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for j in range(points):
        for i in range(points):
            g[i, j, k, 0] = gx[i, j, k] 
            g[i, j, k, 1] = gy[i, j, k]
            g[i, j, k, 2] = gz[i, j, k] 

# g - g' space
dpoints = 4*N + 1            
dNrange = np.arange(-2*N, 2*N+1)
drfreq = 0.5*rfreq
norm_factor = 1.0
while (norm_factor < dpoints):
    norm_factor *= 2.0

[dg_x, dg_y, dg_z] = np.meshgrid(gfreq * dNrange, gfreq * dNrange, gfreq * dNrange)
[dx, dy, dz] = np.meshgrid(drfreq * dNrange, drfreq * dNrange, drfreq * dNrange) 

dr = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            dr[i, j, k, 0] = dx[i, j, k] 
            dr[i, j, k, 1] = dy[i, j, k]
            dr[i, j, k, 2] = dz[i, j, k] 

dg = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            dg[i, j, k, 0] = dg_x[i, j, k] 
            dg[i, j, k, 1] = dg_y[i, j, k]
            dg[i, j, k, 2] = dg_z[i, j, k] 



pore_dr = np.array(np.zeros([dpoints, dpoints, dpoints]))
matrix_dr = np.array(np.zeros([dpoints, dpoints, dpoints]))
for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            px = dx[i,j,k]
            py = dy[i,j,k]
            pz = dz[i,j,k]
            pore_dr[i,j,k] = pore_function(px, py, pz, R)

for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            px = dx[i,j,k]
            py = dy[i,j,k]
            pz = dz[i,j,k]
            matrix_dr[i,j,k] = matrix_function(px, py, pz, R)

# Apply FFT-3D to characteristic signal
pore_dg = apply_fft(pore_dr)
pore_dg = norm_factor * (1.0/volume) * pore_dg
matrix_dg = apply_fft(matrix_dr)
matrix_dg = norm_factor * (1.0/volume) * matrix_dg

pore_dg2 = apply_dft(pore_dr, dr, dg, volume, dpoints)
matrix_dg2 = apply_identity(pore_dg2, dpoints)

# Assembly matrices
rows = points**3
cols = rows
Wdg = np.asmatrix(np.zeros([rows, cols], dtype=complex))
Udg = np.asmatrix(np.zeros([rows, cols], dtype=complex))
Tdg = np.asmatrix(np.zeros([rows, cols], dtype=complex))

# W matrix
for k in range(points):
    for j in range(points):
        for i in range(points):
            row_index = IDX2C_3D(i, j, k, points)
            
            for kk in range(points):
                for jj in range(points):
                    for ii in range(points):
                        col_index = IDX2C_3D(ii, jj, kk, points)
                        di = map_g_to_dg(i,ii,N)
                        dj = map_g_to_dg(j,jj,N)
                        dk = map_g_to_dg(k,kk,N)
                        Wdg[row_index, col_index] = (-1.0) * w * matrix_dg2[di, dj, dk]

for row in range(rows):
    Wdg[row, row] += 1.0

# Get cholesky decomposition R matrix
R = np.linalg.cholesky(Wdg)
Rinv = np.linalg.inv(R)

# U matrix
for k in range(points):
    for j in range(points):
        for i in range(points):
            row_index = IDX2C_3D(i, j, k, points)
            
            for kk in range(points):
                for jj in range(points):
                    for ii in range(points):
                        col_index = IDX2C_3D(ii, jj, kk, points)
                        di = map_g_to_dg(i,ii,N)
                        dj = map_g_to_dg(j,jj,N)
                        dk = map_g_to_dg(k,kk,N)
                        Udg[row_index, col_index] = (-1.0) * w * matrix_dg2[di, dj, dk]

for row in range(rows):
    Udg[row, row] += 1.0
                        
# Solve for q
for q in q_array:
    
    # T matrix
    for k in range(points):
        for j in range(points):
            for i in range(points):
                row_index = IDX2C_3D(i, j, k, points)
                
                for kk in range(points):
                    for jj in range(points):
                        for ii in range(points):
                            col_index = IDX2C_3D(ii, jj, kk, points)
                            Tdg[row_index, col_index] = np.dot((q + g[i,j,k]), (q + g[ii,jj,kk])) * Udg[row_index, col_index] 

    # V matrix
    Vdg = Rinv.H * Tdg * Rinv
    vals, vecs = np.linalg.eig(Vdg)
    vec = vecs.transpose()
    # Sort eigen values and its vector
    inds = np.argsort(vals)
    vals = vals[inds]
    vecs = vecs[inds] 
    weights = (1.0/w) * (R.H - ((1-w) * Rinv)) * (vecs)
    