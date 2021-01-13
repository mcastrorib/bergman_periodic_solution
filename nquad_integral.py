import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt

def func(x, y, z, R):
    d = np.sqrt((x**2 + y**2 + z**2))
    if(d < R):
        return 1.0
    elif(d == R):
        return 0.5
    else:
        return 0.0

def func2(x, y, z, R, g):
    return func(x, y, z, R) * np.cos(g[0] * x + g[1] * y + g[2] * z)

def func3(x,y,z):
    Radius = 50.0
    N = 5
    freq = N * (2.0 * np.pi) / Length
    g_wavevec = np.array([freq, 0, 0]) 
    return func2(x, y, z, Radius, g_wavevec)
    # return x**2 * y**2 * z**2

Length = 10.0
A = (-0.5) * Length
B = (0.5) * Length
Radius = 50.0
N = 5
freq = N * (2.0 * np.pi) / Length
g_wavevec = np.array([freq, 0, 0]) 

# Integrate via scipy module
opts_dict = {
    "epsabs":.5e-01, 
    "epsrel":.5e-01,
    "limit":5
    }

spI = integrate.nquad(func3, [[A, B], [A, B], [A, B]], opts=opts_dict)
# spI = integrate.tplquad(func3, A, B, lambda x: A, lambda x: B, lambda x, y: A, lambda x, y: B)
# spI = 1.0

# Plot
# Generate data for plot
pts = 10
dev = 0.5*Length/pts
x = np.linspace(A+dev, B+dev, pts, False)
y = np.linspace(A+dev, B+dev, pts, False)
z = np.linspace(A+dev, B+dev, pts, False)
dx = x[1] - x[0]
dy = dx
dz = dx
dV = dx**3

Fx = np.zeros([pts,pts,pts])
for k in range(pts):
    for j in range(pts):
        for i in range(pts):
            Fx[i][j][k] = func3(x[i], y[j], z[k])#, Radius, g_wavevec)

intFx = 0.0
for k in range(pts):
    for j in range(pts):
        for i in range(pts):
            intFx += dV * Fx[i][j][k]

print("Integral (expected) = ", (4.0*np.pi*(Radius**3)/3.0))
print("Integral (cubic elementar volumes) = ", intFx)
print("Integral (scipy) = ", spI)