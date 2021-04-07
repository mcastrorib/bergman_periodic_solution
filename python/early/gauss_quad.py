import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt

def func(x, R):
    d = np.sqrt(x**2)
    if(d < R):
        return 1.0
    elif(d == R):
        return 0.5
    else:
        return 0.0

def func2(x, R):
    return x**3 + 2*(x**2)

def func3(x, R, freq):
    return func(x, R) * np.cos(freq * x)

def convert_t_to_x(t, A, B):
    return (0.5 * ((B-A) * t + A + B))

def get_gauss_points_and_weights(N):
    p, w = np.polynomial.legendre.leggauss(N)
    return p, w

Length = 10.0
A = (-0.5) * Length
B = (0.5) * Length
Radius = 2.5
Points = 50
N = 10
freq = 1 * (2.0 * np.pi) / Length

# Get Gauss quadrature points
gp, gw = get_gauss_points_and_weights(Points)

# Convert integral points
X = np.zeros(Points)
for i in range(Points):
    X[i] = convert_t_to_x(gp[i], A, B) 

# Evaluate points
fX = np.zeros(Points)
for i in range(Points):
    fX[i] = func(X[i], Radius) * np.cos(freq * X[i])

# Integrate
fSum = 0.0
J = 0.5 * Length
for i in range(Points):
    fSum += gw[i] * fX[i] * J

# Integrate via scipy module
spI = integrate.quad(lambda x: func3(x, Radius, freq), A, B)

# Integrate via FFT
lk = 2*N + 1
kx = np.zeros(lk)
kmax = N/Length
dk = kmax / ((float(lk//2) + 1.0) - 1.0)
kx[0] = - kmax * 2.0 * np.pi
for i in range(1, lk):
    kx[i] = kx[i-1] + 2.0*np.pi*dk 

x = np.zeros(lk)
dx = 1.0 / (2.0 * kmax)
x[0] = (-0.5) * Length
for i in range(1, lk):
    x[i] = x[i-1] + dx

theta = np.zeros(lk)
for i in range(lk):
    theta[i] = func(x[i], Radius)

ktheta = np.fft.fft(theta)
ktheta = np.fft.fftshift(ktheta)

# Plot
# Generate data for plot
xi = np.linspace(A, B, 1000)
dx = xi[1] - xi[0]
Fxi = np.zeros(1000)
for i in range(1000):
    Fxi[i] = func3(xi[i], Radius, freq)

plt.plot(xi, Fxi)
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.show()

print("gauss weights = ", gw)
print("gauss points = ", gp)
print("X = ", X)
print("f(X) = ", fX)
print("Integral (expected) = ", sum(dx*Fxi))
print("Integral (scipy) = ", spI)
print("Integral (gauss quad) = ", fSum)
print("Integral (FFT) = ", np.real(ktheta[-1]))