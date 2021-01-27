import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def func2(x):
    return np.round(np.random.random())

def func(x,y,z,r):
    l = np.linalg.norm(np.array([x,y,z]))
    if(l < r):
        return 1.0
    else:
        return 0.0


def normalize_signal_1d(signal):
    nsignal = np.zeros(signal.shape[0], dtype=complex)
    s0 = signal.max()
    for i in range(signal.shape[0]):
        nsignal[i] = signal[i]/s0
    return nsignal

def normalize_signal_2d(signal):
    nsignal = np.zeros([signal.shape[0], signal.shape[1]], dtype=complex)
    s0 = signal.max()
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            nsignal[i,j] = signal[i,j]/s0
    return nsignal

def normalize_signal_3d(signal):
    nsignal = np.zeros([signal.shape[0], signal.shape[1], signal.shape[2]], dtype=complex)
    s0 = signal.max()

    for k in range(signal.shape[2]):
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                nsignal[i,j,k] = signal[i,j,k]/s0
    return nsignal

def plot_fft_1d_results(vecX, signal, vecK, dft, fft):       
    points = vecX.shape[0]
    
    fig, axs = plt.subplots(6, 1)
    axs[0].plot(vecX, signal,'o')
    axs[1].plot(vecK, np.real(dft),'-')
    axs[1].plot(vecK, np.imag(dft),'--')
    axs[2].plot(vecK, np.real(fft),'-')
    axs[2].plot(vecK, np.imag(fft),'--')
    axs[3].plot(vecK, np.abs(dft),'-')
    axs[4].plot(vecK, np.abs(fft),'-')
    axs[5].plot(vecK, np.abs(dft),'-')
    axs[5].plot(vecK, np.abs(fft),'--')
    plt.show()
    return

def plot_fft_2d_results(signal, dft, fft):  
    diff_abs = np.abs(dft)-np.abs(fft)
    diff_real = np.real(dft)-np.real(fft)
    diff_imag = np.imag(dft)-np.imag(fft)
    cmap = cm.PRGn
    cmap=cm.get_cmap(cmap)

    points = signal.shape[0]
    fig, axs = plt.subplots(3, 2)
    im00 = axs[0,0].imshow(np.abs(signal), cmap=cmap)
    im10 = axs[1,0].imshow(np.abs(dft), cmap=cmap)
    im20 = axs[2,0].imshow(np.abs(fft), cmap=cmap)
    im01 = axs[0,1].imshow(diff_real, cmap=cmap)
    im11 = axs[1,1].imshow(diff_imag, cmap=cmap)
    im21 = axs[2,1].imshow(diff_abs, cmap=cmap)
    
    axs[0,0].set_title("signal")
    axs[1,0].set_title("dft")
    axs[2,0].set_title("fft")
    axs[0,1].set_title("real(diff)")
    axs[1,1].set_title("imag(diff)")
    axs[2,1].set_title("abs(diff)")
    

    fig.colorbar(im00, ax=axs[0,0])
    fig.colorbar(im10, ax=axs[1,0])
    fig.colorbar(im20, ax=axs[2,0])
    fig.colorbar(im01, ax=axs[0,1])
    fig.colorbar(im11, ax=axs[1,1])
    fig.colorbar(im21, ax=axs[2,1])
    fig.tight_layout()
    plt.show()
    return

def plot_fft_3d_results(signal, dft, fft, nimgs=1):  
    diff_abs = np.abs(dft)-np.abs(fft)
    diff_real = np.real(dft)-np.real(fft)
    diff_imag = np.imag(dft)-np.imag(fft)
    cmap = cm.PRGn
    cmap=cm.get_cmap(cmap)

    points = signal.shape[0]
    fig, axs = plt.subplots(nimgs, 6)
    img_list = []
    for im in range(nimgs):
        im00 = axs[im,0].imshow(np.abs(signal[im]), cmap=cmap)
        im10 = axs[im,1].imshow(np.abs(dft[im]), cmap=cmap)
        im20 = axs[im,2].imshow(np.abs(fft[im]), cmap=cmap)
        im01 = axs[im,3].imshow(diff_real[im], cmap=cmap)
        im11 = axs[im,4].imshow(diff_imag[im], cmap=cmap)
        im21 = axs[im,5].imshow(diff_abs[im], cmap=cmap)             

        fig.colorbar(im00, ax=axs[im,0])
        fig.colorbar(im10, ax=axs[im,1])
        fig.colorbar(im20, ax=axs[im,2])
        fig.colorbar(im01, ax=axs[im,3])
        fig.colorbar(im11, ax=axs[im,4])
        fig.colorbar(im21, ax=axs[im,5])
    
    axs[0,0].set_title("signal")
    axs[0,1].set_title("dft")
    axs[0,2].set_title("fft")
    axs[0,3].set_title("real(diff)")
    axs[0,4].set_title("imag(diff)")
    axs[0,5].set_title("abs(diff)")
    
    for im in range(nimgs):
        for col in range(6):
            axs[im,col].grid(False)

            # Hide axes ticks
            axs[im,col].set_xticks([])
            axs[im,col].set_yticks([])
    
    # fig.tight_layout()
    plt.show()
    return

def apply_dft_1d(signal, vecx, veck, length, points):
    kspec = np.zeros(points, dtype=complex)
    dX = length / (points)
    for i in range(points):
        gsum = 0.0
        for rx in range(points):
            gsum += dX * signal[rx] * np.exp((-1.0j) * veck[i] * vecx[rx])        
        kspec[i] = (1.0 / points) * gsum
    
    return kspec

def apply_dft_2d(signal, vecx, vecy, veckx, vecky, area, points):
    kspec = np.zeros([points, points], dtype=complex)
    dA = area / (points**2)
    for i in range(points):
        for j in range(points):   
            gsum = 0.0
            for ry in range(points):
                for rx in range(points):
                    gsum += dA * signal[ry,rx] * np.exp((-1.0j) * (veckx[j] * vecx[rx] + vecky[i] * vecy[ry]))        
            kspec[i,j] = (1.0 / area) * gsum
    
    return kspec

def apply_dft_3d(signal, vecx, vecy, vecz, veckx, vecky, veckz, volume, points):
    kspec = np.zeros([points, points, points], dtype=complex)
    dV = volume / (points**3)
    elems = points**3
    count = 0
    for k in range(points):
        for i in range(points):
            for j in range(points):   
                count += 1
                print(":: {} fourier coefficient out of {}.".format(count, elems))

                gsum = 0.0
                for rz in range(points):
                    for ry in range(points):
                        for rx in range(points):
                            gsum += dV * signal[ry,rx,rz] * np.exp((-1.0j) * (veckx[j] * vecx[rx] + vecky[i] * vecy[ry] + veckz[k] * vecz[rz]))        
                kspec[i,j, k] = (1.0 / volume) * gsum
        
    return kspec

def apply_fft_1d(signal):
    kspec = np.fft.fft(signal, norm='ortho')
    kspec = np.fft.fftshift(kspec)
    return kspec

def apply_fft_2d(signal):
    kspec = np.fft.fft2(signal, norm='ortho')
    kspec = np.fft.fftshift(kspec)
    return kspec

def apply_fft_3d(signal):
    kspec = np.fft.fftn(signal, norm='ortho')
    kspec = np.fft.fftshift(kspec)
    return kspec

def test_fft1D():
    N = 256
    a = 1.0
    size = 2*N + 1 
    signal = np.zeros(size)

    Xfreq = 0.5*a / float(N)
    # [x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 
    vecX = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]


    Kfreq = 2*np.pi/a
    [vecK] = np.meshgrid(Kfreq*np.arange(-N, N+1))

    # for i in range(size):
    #     signal[i] = func(vecX[i])

    for i in range(size//4):
        signal[size//2-i] = 1.0
        signal[size//2+i] = 1.0
        

    dft_kspec = apply_dft_1d(signal, vecX, vecK, a, size)
    fft_kspec = apply_fft_1d(signal)
    plot_fft_1d_results(vecX, signal, vecK, dft_kspec, fft_kspec)

    norm_fft_kspec = normalize_signal(fft_kspec)
    norm_dft_kspec = normalize_signal(dft_kspec)
    plot_fft_1d_results(vecX, signal, vecK, norm_dft_kspec, norm_fft_kspec)
    return

def test_fft2D():
    N = 16
    a = 1.0
    area = a**2
    size = 2*N + 1 
    signal =  np.zeros([size, size])

    Xfreq = 0.5*a / float(N)
    # [x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 
    vecX = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]
    vecY = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]


    Kfreq = 2*np.pi/a
    [vecKX] = np.meshgrid(Kfreq*np.arange(-N, N+1))
    [vecKY] = np.meshgrid(Kfreq*np.arange(-N, N+1))

    for i in range(size):
        for j in range(size):
            signal[i,j] = func(vecX[i])

    for i in range(size//4):
        for j in range(size//4):
            signal[size//2-i, size//2-j] = 1.0
            signal[size//2+i, size//2-j] = 1.0
        

    dft_kspec = apply_dft_2d(signal, vecX, vecY, vecKX, vecKY, area, size)
    fft_kspec = apply_fft_2d(signal)
    plot_fft_2d_results(signal, dft_kspec, fft_kspec)

    norm_fft_kspec = normalize_signal_2d(fft_kspec)
    norm_dft_kspec = normalize_signal_2d(dft_kspec)
    plot_fft_2d_results(signal, norm_dft_kspec, norm_fft_kspec)
    return

N = 6
a = 1.0
volume = a**3
size = 2*N + 1 
signal =  np.zeros([size, size, size])

Xfreq = 0.5*a / float(N)
# [x, y, z] = np.meshgrid(rfreq * Nrange, rfreq * Nrange, rfreq * Nrange) 
vecX = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]
vecY = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]
vecZ = np.linspace(-0.5*a, 0.5*a, 2*size + 1)[1:2*size:2]


Kfreq = 2*np.pi/a
[vecKX] = np.meshgrid(Kfreq*np.arange(-N, N+1))
[vecKY] = np.meshgrid(Kfreq*np.arange(-N, N+1))
[vecKZ] = np.meshgrid(Kfreq*np.arange(-N, N+1))

for k in range(size):
    for i in range(size):
        for j in range(size):
            signal[i,j, k] = func(vecX[j], vecY[i], vecZ[k], a/2)

for k in range(size//4):
    for i in range(size//4):
        for j in range(size//4):
            signal[size//2-i, size//2-j, size//2-k] = 1.0
            signal[size//2+i, size//2-j, size//2+k] = 1.0
    

dft_kspec = apply_dft_3d(signal, vecX, vecY, vecZ, vecKX, vecKY, vecKZ, volume, size)
fft_kspec = apply_fft_3d(signal)
plot_fft_3d_results(signal, dft_kspec, fft_kspec, size)

norm_fft_kspec = normalize_signal_3d(fft_kspec)
norm_dft_kspec = normalize_signal_3d(dft_kspec)
plot_fft_3d_results(signal, norm_dft_kspec, norm_fft_kspec, size)