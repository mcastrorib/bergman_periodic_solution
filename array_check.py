import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def checkIfListContainsArray(_list, _array, _size, _dim):
    for idx in range(_size):
            element = _list[idx]
            coord = 0
            equal = True
            while(coord < _dim and equal == True):
                if(element[coord] != _array[coord]):
                    equal = False
                coord += 1

            if(equal == True and coord == _dim):
                return True
    
    return False

def gain(N):
    upper = ((4*N + 1)**3 + 1)
    lower = ((2*N + 1)**3) * (((2*N + 1)**3 + 1))
    # print("gain = {:.3f} / {:.3f}".format(upper, lower))
    return upper/lower 

N = 1
Nvec_length = int(2*N + 1)
dimension = 3
points = Nvec_length**(dimension)
Gvec = np.zeros([points, dimension])
Nvec = np.linspace(-N, N, Nvec_length)

row = 0
if(dimension == 1):
    for x in range(Nvec_length):
        Gvec[row][0] = x
        row += 1
elif(dimension == 2):
    for y in range(Nvec_length):
        for x in range(Nvec_length):
            Gvec[row][0] = x
            Gvec[row][1] = y
            row += 1
elif(dimension == 3):
    for z in range(Nvec_length):
        for y in range(Nvec_length):
            for x in range(Nvec_length):
                Gvec[row][0] = x
                Gvec[row][1] = y
                Gvec[row][2] = z
                row += 1
              

Gdif = []
for row in range(points):
    print("Gdif:: building row", row, "out of", points)
    line = []
    for col in range(points):
        line.append(Gvec[row] - Gvec[col])
    Gdif.append(line)

# print(Gdif)

Glist = []
Glist_size = 0
Gnewcase = np.asmatrix(np.zeros([points, points], dtype=bool))
for row in range(points):
    print("Gnewcase:: building row", row, "out of", points, " - Glist size: ", Glist_size)
    for col in range(points):
        Gnew = Gdif[row][col]
        Gnew2 = (-1) * Gnew
        if(checkIfListContainsArray(Glist, Gnew, Glist_size, dimension) or checkIfListContainsArray(Glist, Gnew2, Glist_size, dimension)):
            Gnewcase[row, col] = False
        else:
            Gnewcase[row, col] = True
            Glist.append(Gnew)
            Glist_size += 1



x = np.arange(1, 11, 1)
gains = np.zeros(x.shape[0])
for idx in range(x.shape[0]):
    gains[idx] = gain(x[idx])