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
    
    print("k = ", kvec)
    print(gk_index, gkvec, distance)
    print(kxmin, kxmax, kxinc)
    print(kymin, kymax, kyinc)
    print(kzmin, kzmax, kzinc)
    for k in range(kzmin, kzmax, kzinc):
        for j in range(kymin, kymax, kyinc):
            for i in range(kxmin, kxmax, kxinc):
                
                diffvec = kvec - g[j, i, k]
                new_distance = np.sqrt(np.dot(diffvec, diffvec))
                print("i,j,k = ", i,j,k, "G = ", g[j, i, k], new_distance)

                if(new_distance < distance):
                    distance = new_distance
                    gkvec = g[j, i, k]
                    gk_index = IDX2C_3D(i, j, k, points)
                    print(gk_index, gkvec, distance)             
    
    return gkvec, gk_index
    


# Problem parameters
D_p = 2.5
D_m = 0.0
a = 10.0
R = 5.0
N = int(1)
w = 0.9999
u = 1.0

k_direction = np.array([1,0,0])
ka_max = 0.1
k_points = 5
k_array = np.zeros([k_points, 3])
k_linspace = np.linspace(ka_max/a, 0.0, k_points)
for i in range(3):
    k_array[:, i] = k_direction[i] * k_linspace
 

volume = a**3
porosity = 1.0 + (np.pi / 4.0) - (3.0 * np.pi * (R/a)**2) + ((8.0 / 3.0 ) * np.pi) * (R/a)**3 
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

for k in range(points):
    for i in range(points):
        for j in range(points):
            px = x[i,j,k]
            py = y[i,j,k]
            pz = z[i,j,k]
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

for k in range(dpoints):
    for i in range(dpoints):
        for j in range(dpoints):
            px = dx[i,j,k]
            py = dy[i,j,k]
            pz = dz[i,j,k]
            matrix_dr[i,j,k] = matrix_function(px, py, pz, R)

# Apply FFT-3D to characteristic signal
# pore_dg = apply_fft(pore_dr)
# pore_dg = norm_factor * (1.0/volume) * pore_dg
# matrix_dg = apply_fft(matrix_dr)
# matrix_dg = norm_factor * (1.0/volume) * matrix_dg

pore_dg = apply_dft(pore_dr, dr, dg, volume, dpoints)
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
                        matW[row_index, col_index] = (-1.0) * w * matrix_dg[di, dj, dk]

for row in range(rows):
    matW[row, row] += 1.0

# Get cholesky decomposition R matrix
matRH = np.linalg.cholesky(matW)
matRHinv = np.linalg.inv(matRH)
matR = matRH.H
matRinv = np.linalg.inv(matR)

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
kvec = np.array([0.0, 0.0, 0.0])

# Find gk in reciprocal lattice
gkvec, gk_index = find_gk(kvec, g, points)
qvec = kvec - gkvec

# T matrix
for k in range(points):
    for i in range(points):
        for j in range(points):
            row_index = IDX2C_3D(i, j, k, points)
            
            for kk in range(points):
                for ii in range(points):
                    for jj in range(points):
                        col_index = IDX2C_3D(ii, jj, kk, points)
                        matT[row_index, col_index] = np.dot((qvec + g[i,j,k]), (qvec + g[ii,jj,kk])) * matU[row_index, col_index] 

# V matrix
matV = matRinv.H * matT * matRinv
vals, vecs = np.linalg.eig(matW)
vec = vecs.transpose()
# Sort eigen values and its vector
inds = np.argsort(vals)
vals = vals[inds]
vecs = vecs[inds] 
weights = (1.0/w) * (matRH - ((1-w) * matRinv)) * (vecs)

# M(k,t)
time = 10.0
Mkt = 0.0
Mkt_sum = 0.0
for n in range(points**3):
    Mkt_sum += np.exp((-1.0) * vals[n] * time) * (np.abs(weights[gk_index, n]))**2
Mkt = (1.0 / porosity) * Mkt_sum

quantity = np.zeros(rows, dtype=complex)
matAux = np.asmatrix(np.zeros([rows, cols], dtype=complex))
for n in range(vals.shape[-1]):
    matAux = (matR * matRH)
    value = ((vecs[n] * matAux) * vecs[n].H)
    quantity[n] = value[0,0]

reduced_vals = (a**2 / D_p) * vals