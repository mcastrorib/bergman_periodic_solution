import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def check_pore_region(_x, _y, _z, _radius):
    distance = np.sqrt((_x * _x) + (_y * _y) + (_z * _z))
    if(distance > _radius):
        return True
    else:
        return False

def check_matrix_region(_x, _y, _z, _radius):
    distance = np.sqrt((_x * _x) + (_y * _y) + (_z * _z))
    if(distance > _radius):
        return False
    else:
        return True 

def getTggMatrix(_G, _Dp, _q, _theta_Pr):
    n_points = _theta_Pr.shape[0]
    Tgg = np.asmatrix(np.zeros([n_points, n_points]), dtype=complex)
    
    for g_idx in range(n_points):
        for gl_idx in range(n_points):
            Pr_idx = _G[gl_idx] - _G[g_idx]
            Tgg[g, gline] = _Dp * np.dot((_G[g_idx] + _q),(_G[gl_idx] + _q)) * _theta_Pr[Pr_idx]
    
    return Tgg


# -- Essential parameters
# [LENGTH] = um
# [TIME] = ms
D_p = 2.5
unit_cell_length = 10.0
unit_cell_volume = unit_cell_length**3
sphere_radius = 5.0
unit_cell_divisions = 1
n_length = int(1 + (2 * unit_cell_divisions))
n_points = n_length**3

n_x = np.linspace(-unit_cell_divisions, unit_cell_divisions, n_length)
n_y = np.linspace(-unit_cell_divisions, unit_cell_divisions, n_length)
n_z = np.linspace(-unit_cell_divisions, unit_cell_divisions, n_length)

# --Build spatial vector R
r_gap = (0.5 * unit_cell_length) / unit_cell_divisions
r_max = unit_cell_divisions * r_gap
r_min = (-1) * r_max
r_x = np.linspace(r_min, r_max, n_length)
r_y = np.linspace(r_min, r_max, n_length)
r_z = np.linspace(r_min, r_max, n_length)
R = np.zeros([n_points, 3])
idx = 0
for z in r_z:
    for y in r_y:
        for x in r_x:
            R[idx, 0] = x
            R[idx, 1] = y
            R[idx, 2] = z
            idx += 1

# -- Build wavevector G
angular_frequency = (2 * np.pi) / unit_cell_length
G = np.zeros([n_points, 3])
idx = 0
for z in n_z:
    for y in n_y:
        for x in n_x:
            G[idx, 0] = angular_frequency * x
            G[idx, 1] = angular_frequency * y
            G[idx, 2] = angular_frequency * z
            idx += 1


# -- Build characteristic spatial function 
# Matrix region
theta_Mr = np.zeros(n_points)
for idx in range(n_points):
    coord_x = R[idx, 0]
    coord_y = R[idx, 1]
    coord_z = R[idx, 2]
    if(check_matrix_region(coord_x, coord_y, coord_z, sphere_radius)):
        theta_Mr[idx] = 1.0
    else:
        theta_Mr[idx] = 0.

# Pore region
theta_Pr = np.zeros(n_points)
for idx in range(n_points):
    coord_x = R[idx, 0]
    coord_y = R[idx, 1]
    coord_z = R[idx, 2]
    if(check_pore_region(coord_x, coord_y, coord_z, sphere_radius)):
        theta_Pr[idx] = 1.0
    else:
        theta_Pr[idx] = 0.0


# -- Build characteristic wavevector functions
# Matrix region
theta_Mg = np.zeros(n_points, dtype=complex)
for G_idx in range(n_points):
    print("theta_Mg idx: ", G_idx)
    new_entry = 0.0 + 0.0j
    Gvalue = G[G_idx]

    for R_idx in range(n_points):
        Rvalue = R[R_idx]
        theta_value = theta_Mr[R_idx]
        new_entry +=  theta_value * np.exp((-1.0j) * np.dot(Gvalue, Rvalue))
        
    theta_Mg[G_idx] = ( new_entry / unit_cell_volume )
    
# Pore region - defining as theta_Pr[g] = delta(g0) - theta_Mr(g)
theta_Pg = np.zeros(n_points, dtype=complex)
for G_idx in range(n_points):
    print("theta_Pg idx: ", G_idx)
    theta_Pg[G_idx] = (-1)*theta_Mg[G_idx]
theta_Pg[n_points//2] += 1.0

# -- Build eigenvalue problem matrices
Tgg = np.asmatrix(np.zeros([n_points, n_points], dtype=complex))
Wgg = np.asmatrix(np.zeros([n_points, n_points], dtype=complex))



# -- Datavis for debug
fig = plt.figure(figsize=plt.figaspect(1.0), dpi=100)
axis_lims = [-0.55*unit_cell_length, 0.55*unit_cell_length]
viewport = [axis_lims, axis_lims, axis_lims]
ax = fig.add_subplot(111, projection='3d')

# Points (s=point_size, c=color, cmap=colormap)
data_x = [] 
data_y = [] 
data_z = [] 
for idx in range(n_points):
    if(theta_Mr[idx] == 1):
        data_x.append(R[idx, 0])
        data_y.append(R[idx, 1])
        data_z.append(R[idx, 2])
        
ax.scatter(data_x, data_y, data_z, zdir='y', s=10.0, c='black', marker='o', alpha=1.0) 
ax.set_title('')
ax.set_xlim(viewport[0])
ax.set_ylim(viewport[2])
ax.set_zlim(viewport[1])
ax.grid(False) 
# plt.show()