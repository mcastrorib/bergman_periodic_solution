import numpy as np
import matplotlib
import matplotlib.pyplot as plt

a = 1.0
R = 0.50
N = int(3)

volume = a**3
points = 2*N + 1
Nrange = np.arange(-N, N+1)

rfreq = 0.5*a / float(N)
x = np.meshgrid(rfreq * Nrange) 

gfreq = 2*np.pi/a
gx = np.meshgrid(gfreq*Nrange)

fx = np.zeros(points)
for i in range(points):
    fx[i] = np.cos(4*np.pi*x[0][i])
    # fx[i] = 1.0

print("freq = ", gx[0][3])

kfx = np.fft.fft(fx)
kfx = np.fft.fftshift(kfx)
kfreq = np.fft.fftshift(np.fft.fftfreq(x[0].shape[-1]))

kfx2 = np.zeros(points, dtype=complex)
for i in range(points):
    ksum = 0.0
    for j in range(points):
        # print("j =", j, "\t f[",j,"] = ",fx[0,j])
        ksum += fx[j] * np.exp((-1.0j) * gx[0][i] * x[0][j])
    kfx2[i] = ksum

dk = kfreq[1] - kfreq[0]
lx = points
kmax = dk * ((lx//2 + 1) - 1)
dx = 1.0 / (2.0 * kmax)
x3 = np.zeros(points)
x3[0] = x[0][0]
for i in range(1, points):
    x3[i] = x3[i-1] + dx

fx3 = np.zeros(points)
for i in range(1, points):
    fx3[i] = np.cos(gx[0][3]*x[0][i])



plt.plot(x[0], fx)
plt.show()

plt.plot(gx[0], np.abs(np.real(kfx)), color="blue")
plt.plot(gx[0], np.abs(np.real(kfx2)), '--', color="red")
plt.show()

plt.plot(gx[0], np.imag(kfx), color="blue")
plt.plot(gx[0], np.imag(kfx2), '--', color="red")
plt.show()