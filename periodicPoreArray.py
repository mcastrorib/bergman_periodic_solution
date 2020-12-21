import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class PeriodicPoreArray():
    def __init__(self, _unit_cell_length=1.0, _unit_cell_divisions=1, _grain_radius=0.5, _fluid_diffusion_coefficient=2.5, _omega=0.9999):
        # -- Initialize essentials
        # Public parameters
        self.unit_cell_length = _unit_cell_length
        self.grain_radius = _grain_radius
        self.unit_cell_divisions = _unit_cell_divisions
        self.fluid_diffusion_coefficient = _fluid_diffusion_coefficient
        self.omega = _omega

        # Private parameters
        self._unit_cell_volume = _unit_cell_length**3
        self._n_linspace_length = int(1 + (2 * _unit_cell_divisions))
        self._n_linspace = np.linspace(-_unit_cell_divisions, _unit_cell_divisions, self.get_n_linspace_length())
        self._points = self.get_n_linspace_length()**3
        
        self._essentials = True

        # -- Initialize vectors and matrices
        # Spatial vector R
        self.R = np.zeros([self.get_points(), 3])

        # Wavevector G
        self.G = np.zeros([self.get_points(), 3])

        # Characteristic spatial functions 
        # Pr: pore 
        # Mr: matrix 
        self.theta_Pr = np.zeros(self.get_points())
        self.theta_Mr = np.zeros(self.get_points())

        # Eigenvalue problem matrices
        self.Tgg = np.asmatrix(np.zeros([self.get_points(), self.get_points()], dtype=complex))
        self.Wgg = np.asmatrix(np.zeros([self.get_points(), self.get_points()], dtype=complex))        
        
        return
    
    # -- Build up
    def build(self):
        if(self.get_essentials() == False):
            self.build_essentials()
        
        self.build_spatial_vector_R()
        self.build_wave_vector_G()
        self.build_characteristic_spatial_pore_function()
        self.build_characteristic_spatial_matrix_function()
        return
    
    def build_essentials(self):
        self.set_unit_cell_volume()
        self.set_n_linspace_length()
        self.set_n_linspace()
        self.set_points()
        self.set_essentials(True)
        return
    
    def build_spatial_vector_R(self):
        self.R = np.zeros([self.get_points(), 3])
        
        r_gap = (0.5 * self.get_unit_cell_length()) / self.get_unit_cell_divisions()
        r_max = self.get_unit_cell_divisions() * r_gap
        r_min = (-1) * r_max
        r_array = np.linspace(r_min, r_max, self.get_n_linspace_length())
        
        idx = 0
        for z in r_array:
            for y in r_array:
                for x in r_array:
                    self.set_R(idx, x, y, z)
                    idx += 1
        
        return
    
    def build_wave_vector_G(self):
        self.G = np.zeros([self.get_points(), 3])
        
        n_array = self.get_n_linspace()
        angular_frequency = (2 * np.pi) / self.get_unit_cell_length()
        
        idx = 0
        for kz in n_array:
            for ky in n_array:
                for kx in n_array:
                    self.set_G(idx, angular_frequency * kx, angular_frequency * ky, angular_frequency * kz)
                    idx += 1
        
        return
    
    def build_characteristic_spatial_pore_function(self):
        self.theta_Pr = np.zeros(self.get_points())
        n_points = self.get_points()
        for idx in range(n_points):
            coord_x = self.get_R(idx, 0)
            coord_y = self.get_R(idx, 1)
            coord_z = self.get_R(idx, 2)
            if(self.check_pore_region(coord_x, coord_y, coord_z)):
                self.theta_Pr[idx] = 1.0
            else:
                self.theta_Pr[idx] = 0.0
    
    def check_pore_region(self, _x, _y, _z):
        distance = np.sqrt((_x * _x) + (_y * _y) + (_z * _z))
        if(distance > self.get_grain_radius()):
            return True
        else:
            return False
    
    def build_characteristic_spatial_matrix_function(self):
        self.theta_Mr = np.zeros(self.get_points())
        n_points = self.get_points()
        for idx in range(n_points):
            coord_x = self.get_R(idx, 0)
            coord_y = self.get_R(idx, 1)
            coord_z = self.get_R(idx, 2)
            if(self.check_matrix_region(coord_x, coord_y, coord_z)):
                self.theta_Mr[idx] = 1.0
            else:
                self.theta_Mr[idx] = 0.0

    def check_matrix_region(self, _x, _y, _z):
        distance = np.sqrt((_x * _x) + (_y * _y) + (_z * _z))
        if(distance > self.get_grain_radius()):
            return False
        else:
            return True    

    # -- set and get methods
    def set_essentials(self, _bvalue):
        self._essentials = _bvalue
        return
    
    def get_essentials(self):
        return self._essentials

    def set_unit_cell_length(self, _a):
        self.unit_cell_length = _a
        self.set_essentials(False)
        return
    
    def get_unit_cell_length(self):
        return self.unit_cell_length
    
    def set_grain_radius(self, _r):
        self.grain_radius = _r
        self.set_essentials(False)
        return

    def get_grain_radius(self):
        return self.grain_radius
    
    def set_unit_cell_divisions(self, _N):
        self.unit_cell_divisions = _N
        self.set_essentials(False)
        return
    
    def get_unit_cell_divisions(self):
        return self.unit_cell_divisions
    
    def set_fluid_diffusion_coefficient(self, _Dp):
        self.fluid_diffusion_coefficient = _Dp
        self.set_essentials(False)
        return
    
    def get_fluid_diffusion_coefficient(self):
        return self.fluid_diffusion_coefficient
    
    def set_omega(self, _w):
        self.omega = _w
        self.set_essentials(False)
        return
    
    def get_omega(self):
        return self.omega
    
    def set_unit_cell_volume(self):
        a = self.get_unit_cell_length()
        self._unit_cell_volume = a**3
        return
    
    def get_unit_cell_volume(self):
        return self._unit_cell_volume
    
    def set_n_linspace_length(self):
        self._n_linspace_length = int(1 + (2 * self.unit_cell_divisions))
        return
    
    def get_n_linspace_length(self):
        return self._n_linspace_length
    
    def set_n_linspace(self):
        self._n_linspace = np.linspace((-1)*self.get_unit_cell_divisions(), self.get_unit_cell_divisions(), self.get_n_linspace_length())
        return
    
    def get_n_linspace(self):
        return self._n_linspace
    
    def get_n_linspace(self, index=None):
        if(index == None):
            return self._n_linspace
        else:  
            if(index < self.get_n_linspace_length()):
                return self._n_linspace[index]
            else:
                print('error: index is out of range of N linspace array.')
                return
    
    def set_points(self):
        self._points = self.get_n_linspace_length()**3
        return

    def get_points(self):
        return self._points

    def set_R(self, index, x, y, z):
        self.R[index, 0] = x
        self.R[index, 1] = y
        self.R[index, 2] = z
        return
        
    def get_R(self, index=None, coord=None):
        if(index == None):
            return self.R
        elif(coord == None):
            return self.R[index]
        else:
            return self.R[index, coord]
    
    def set_G(self, index, kx, ky, kz):
        self.G[index, 0] = kx
        self.G[index, 1] = ky
        self.G[index, 2] = kz
        return

    def get_G(self, index=None, coord=None):
        if(index == None):
            return self.G
        elif(coord == None):
            return self.G[index]
        else:
            return self.G[index, coord] 
    
    def set_theta_Pr(self, index, value):
        self.theta_Pr[index] = value
        return
    
    def get_theta_Pr(self, index=None):
        if(index == None):
            return self.theta_Pr
        else:
            return self.theta_Pr[index]
    
    def set_theta_Mr(self, index, value):
        self.theta_Mr[index] = value
        return
    
    def get_theta_Mr(self, index=None):
        if(index == None):
            return self.theta_Mr
        else:
            return self.theta_Mr[index]
    
    # -- Datavis for debug
    def dataviz_theta(self):
        fig = plt.figure(figsize=plt.figaspect(1.0), dpi=100)
        axis_lims = [-0.55 * self.get_unit_cell_length(), 0.55 * self.get_unit_cell_length()]
        viewport = [axis_lims, axis_lims, axis_lims]
        ax = fig.add_subplot(111, projection='3d')

        # Points (s=point_size, c=color, cmap=colormap)
        data_x = [] 
        data_y = [] 
        data_z = [] 
        n_points = self.get_points()
        for idx in range(n_points):
            if(self.get_theta_Mr(idx) == 1):
                point = self.get_R(idx)
                data_x.append(point[0])
                data_y.append(point[1])
                data_z.append(point[2])
                
        ax.scatter(data_x, data_y, data_z, zdir='y', s=10.0, c='black', marker='o', alpha=1.0) 
        ax.set_title('')
        ax.set_xlim(viewport[0])
        ax.set_ylim(viewport[2])
        ax.set_zlim(viewport[1])
        ax.grid(False) 
        plt.show()
        return


    
