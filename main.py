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
points = 2*N + 1
Nrange = np.arange(-N, N+1)

rfreq = 0.5*a / float(N)
[x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 

gfreq = 2*np.pi/a
[gx, gy, gz] = np.meshgrid(gfreq*Nrange, gfreq*Nrange, gfreq*Nrange)

r = np.array(np.zeros([points, points, points, 3]))
for k in range(points):
    for j in range(points):
        for i in range(points):
            r[i, j, k, 0] = x[i, j, k] 
            r[i, j, k, 1] = y[i, j, k]
            r[i, j, k, 2] = z[i, j, k] 
            

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
matrix_g = np.fft.fftn(matrix_r)