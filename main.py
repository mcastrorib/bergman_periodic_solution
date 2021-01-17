import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

a = 10.0
R = 5.0
N = int(1)

volume = a**3
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

pore_g = np.fft.fftn(pore_r)
pore_g = np.fft.fftshift(pore_g)
matrix_g = np.fft.fftn(matrix_r)
matrix_g = np.fft.fftshift(matrix_g)

# g - g' space
            
dNrange = np.arange(-2*N, 2*N+1)
drfreq = 0.5*rfreq

[dg_x, dg_y, dg_z] = np.meshgrid(gfreq * dNrange, gfreq * dNrange, gfreq * dNrange)
[dx, dy, dz] = np.meshgrid(drfreq * dNrange, drfreq * dNrange, drfreq * dNrange) 

dpoints = 4*N + 1

r = np.array(np.zeros([dpoints, dpoints, dpoints, 3]))
for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            r[i, j, k, 0] = dx[i, j, k] 
            r[i, j, k, 1] = dy[i, j, k]
            r[i, j, k, 2] = dz[i, j, k] 

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

pore_dg = np.fft.fftn(pore_dr)
pore_dg = np.fft.fftshift(pore_dg)
pore_dg = (1.0/volume) * pore_dg
matrix_dg = np.fft.fftn(matrix_dr)
matrix_dg = np.fft.fftshift(matrix_dg)
matrix_dg = (1.0/volume) * matrix_dg

exp_pdg = np.zeros([dpoints, dpoints, dpoints], dtype=complex)
dV = volume / (dpoints**3)
for k in range(dpoints):
    for j in range(dpoints):
        for i in range(dpoints):
            dG = np.array([dg_x[i,j,k], dg_y[i,j,k], dg_z[i,j,k]])
            gsum = 0.0
            for rz in range(dpoints):
                for ry in range(dpoints):
                    for rx in range(dpoints):
                        gsum += dV * pore_dr[rx, ry, rz] * np.exp((-1.0j) * np.dot(dG, r[rx,ry,rz]))
            
            exp_pdg[i,j,k] = (1.0 / volume) * gsum