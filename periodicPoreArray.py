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
        self._verbose = False
        self._unit_cell_volume = _unit_cell_length**3
        self._n_linspace_length = int(1 + (2 * _unit_cell_divisions))
        self._n_linspace = np.linspace(-_unit_cell_divisions, _unit_cell_divisions, self.get_n_linspace_length())
        self._n_points = self.get_n_linspace_length()**3
        self._porosity = self.set_porosity()
        self._essentials = True

        # -- Initialize vectors and matrices
        # Spatial vector R
        self.spatial_points = 0
        self.spatial_divisions = self.get_unit_cell_divisions()
        self.set_spatial_points()
        self.R = np.zeros([self.get_spatial_points(), 3])

        # Wavevector G
        self.reciprocal_points = 0
        self.reciprocal_divisions = self.get_unit_cell_divisions()
        self.set_reciprocal_points()
        self.G = np.zeros([self.get_reciprocal_points(), 3])

        # Characteristic spatial functions 
        # Pr: pore 
        # Mr: matrix 
        self.theta_Pr = np.zeros(self.get_spatial_points())
        self.theta_Mr = np.zeros(self.get_spatial_points())
        self.theta_Pg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        self.theta_Mg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))

        # Initialize eigenvalue problem matrices
        self.k_points = 1 
        self.ka_max = 2.0
        self.wavevector_k = np.zeros([self.get_k_points(), 3])
        self.wavevector_gk = np.zeros(3) # PODE SER SÓ O INDEX PRO G CORRESPONDENTE!!!
        self.k_index = 0
        self.gk_index = 0
        self.wavevector_q = np.zeros(3)
        self.Tgg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        self.Wgg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        self.invW = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        self.invWxT = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        self.eigen_values = np.zeros(self.get_reciprocal_points(), dtype=complex)
        self.eigen_vectors = np.asmatrix(np.zeros([self.get_reciprocal_points(),self.get_reciprocal_points()], dtype=complex)) 
        self.eigen_states = np.zeros(self.get_reciprocal_points(), dtype=complex)
        self.Mkt = np.zeros(self.get_reciprocal_points(), dtype=complex)
        
        return
    
    # -- Build up
    def build(self):
        if(self.get_essentials() == False):
            self.build_essentials()
        
        self.build_spatial_vector_R()
        self.build_wavevector_G()
        self.build_characteristic_spatial_pore_function_array()
        self.build_characteristic_spatial_matrix_function_array()
        self.build_characteristic_reciprocal_matrix_function_matrix()
        self.build_characteristic_reciprocal_pore_function_matrix()
        return
    
    def build_essentials(self):
        self.set_porosity()
        self.set_unit_cell_volume()
        self.set_n_linspace_length()
        self.set_n_linspace()
        self.set_n_points()
        self.set_spatial_points()
        self.set_reciprocal_points()
        self.set_essentials(True)
        return
    
    def build_spatial_vector_R(self):
        self.R = np.zeros([self.get_spatial_points(), 3])
        
        dr = self.get_unit_cell_length() / self.get_spatial_divisions()
        
        idx = 0
        points = self.get_spatial_divisions()
        x0 = (-0.5) * (self.get_unit_cell_length() - dr)
        y0 = (-0.5) * (self.get_unit_cell_length() - dr)
        z0 = (-0.5) * (self.get_unit_cell_length() - dr)
        for z in range(points):
            for y in range(points):
                for x in range(points):
                    self.set_R(idx, (x0 + x*dr), (y0 + y*dr), (z0 + z*dr))
                    idx += 1
        
        return
    
    def build_wavevector_G(self):
        self.G = np.zeros([self.get_reciprocal_points(), 3])
        
        n_array = self.get_n_linspace()
        angular_frequency = (2 * np.pi) / self.get_unit_cell_length()
        
        idx = 0
        for kz in n_array:
            for ky in n_array:
                for kx in n_array:
                    self.set_G(idx, angular_frequency * kx, angular_frequency * ky, angular_frequency * kz)
                    idx += 1
        
        return
    
    def build_characteristic_spatial_pore_function_array(self):
        self.theta_Pr = np.zeros(self.get_spatial_points())

        n_points = self.get_spatial_points()
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
    
    def build_characteristic_spatial_matrix_function_array(self):
        self.theta_Mr = np.zeros(self.get_spatial_points())

        n_points = self.get_spatial_points()
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
        
    def build_characteristic_reciprocal_pore_function_matrix(self):
        self.theta_Pg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))

        rows = self.get_reciprocal_points()
        cols = self.get_reciprocal_points()
        for row in range(rows):
            if(not self._verbose):
                print("Theta_Pg:: building row", row, "out of", rows)
            for col in range(row, cols):
                Gresult = self.get_G(row) - self.get_G(col)
                # new_theta_Pg = self.get_characteristic_reciprocal_pore_function(Gresult)
                new_theta_Pg = self.get_characteristic_reciprocal_pore_function(row, col)
                self.set_theta_Pg(row, col, new_theta_Pg)
                self.set_theta_Pg(col, row, new_theta_Pg) 

        return
    
    def build_characteristic_reciprocal_matrix_function_matrix(self):
        self.theta_Mg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))

        rows = self.get_reciprocal_points()
        cols = self.get_reciprocal_points()
        for row in range(rows):
            if(not self._verbose):
                print("Theta_Mg:: building row", row, "out of", rows)
            for col in range(row, cols):
                Gresult = self.get_G(row) - self.get_G(col)
                new_theta_Mg = self.get_characteristic_reciprocal_matrix_function(Gresult)
                self.set_theta_Mg(row, col, new_theta_Mg)
                self.set_theta_Mg(col, row, new_theta_Mg)               
        
        return

    def get_characteristic_reciprocal_matrix_function(self, _input1, _input2=None):
        if(type(_input1) == np.ndarray):
            return self._get_characteristic_reciprocal_matrix_function_by_integration_(_input1)
        elif(type(_input1) == int and type(_input2) == int):
            return self._get_characteristic_reciprocal_matrix_function_by_identity_(_input1, _input2)

    
    def get_characteristic_reciprocal_pore_function(self, _input1, _input2=None):
        if(type(_input1) == np.ndarray):
            return self._get_characteristic_reciprocal_pore_function_by_integration_(_input1)
        elif(type(_input1) == int and type(_input2) == int):
            return self._get_characteristic_reciprocal_pore_function_by_identity_(_input1, _input2)
    
    def _get_characteristic_reciprocal_matrix_function_by_integration_(self, _wavevector: np.ndarray):
        # compute theta_Mg -- characteristic matrix function in the reciprocal lattice vector domain 
        theta_Mg = 0.0
        dV = self.get_unit_cell_volume() / self.get_spatial_points()
        for idx in range(self.get_spatial_points()):
            theta_Mg += dV * self.get_theta_Mr(idx) * np.exp(-1j * np.dot(_wavevector, self.get_R(idx)))
        theta_Mg /= self.get_unit_cell_volume()
        
        return np.real(theta_Mg)

    def _get_characteristic_reciprocal_matrix_function_by_identity_(self, _row: int, _col: int):
        # compute theta_Pg -- characteristic pore function in the reciprocal lattice vector domain
        theta_Mg = (-1) * self.theta_Pg[_row, _col]
        if(_row == _col):
            theta_Mg += 1.0       

        return np.real(theta_Mg)    
    
    def _get_characteristic_reciprocal_pore_function_by_integration_(self, _wavevector: np.ndarray):
        # compute theta_Mg -- characteristic matrix function in the reciprocal lattice vector domain 
        theta_Pg = 0.0
        
        dV = self.get_unit_cell_volume() / self.get_spatial_points()
        for idx in range(self.get_spatial_points()):
            theta_Pg += dV * self.get_theta_Pr(idx) * np.exp(-1j * np.dot(_wavevector, self.get_R(idx)))
        theta_Pg /= self.get_unit_cell_volume()
        
        return np.real(theta_Pg)

    def _get_characteristic_reciprocal_pore_function_by_identity_(self, _row: int, _col: int):
        # compute theta_Pg -- characteristic pore function in the reciprocal lattice vector domain
        theta_Pg = (-1) * self.theta_Mg[_row, _col]
        if(_row == _col):
            theta_Pg += 1.0       

        return np.real(theta_Pg)    

    # -- Solve
    def solve(self, _k, _time):
        self.set_wavevector_k(_k)
        self.build_Mkt()
        
        for k_idx in range(self.get_k_points()):
            print("-- K index: ", k_idx)
            self.build_reciprocal_lattice_vectors(k_idx)
            self.build_matrices()
            self.solve_eigenvalues()
            self.build_eigenstates()
            self.solve_Mkt(k_idx, _time)
            self.normalize_Mkt()
        
        return
    
    def build_reciprocal_lattice_vectors(self, _kIdx):
        self.set_k_index(_kIdx)
        print("k =\t {}".format(self.get_wavevector_k(self.get_k_index())))
        self.set_wavevector_gk()
        print("gk =\t {} index = {}".format(self.get_wavevector_gk(), self.get_gk_index()))
        self.set_wavevector_q()
        print("q =\t {} \n".format(self.get_wavevector_q()))
        return

    def build_matrices(self):        
        # - Build Wgg matrix 
        self.build_matrix_Wgg()

        # - Build Tgg matrix
        self.build_matrix_Tgg()
                
        # - Get inverse of Wgg matrix (i.e, Wgg^{-1})
        if(self._verbose):
                print("inverting matrix W...")
        self.build_matrix_invW()

        # - Get product of inv(Wgg) * Tgg
        if(self._verbose):
                print("multiplying matrices inv(W) x T...")
        self.build_matrix_invWxT()
        return
    
    def build_matrix_Tgg(self):
        # - Reinitialize matrix
        self.Tgg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))

        # - Build Tgg matrix
        rows = self.get_reciprocal_points()
        cols = rows
        for gRow in range(rows):
            if(self._verbose):
                print("Tgg matrix:: building row ", gRow, "out of", rows)
            for gCol in range(gRow, cols):
                # - Tgg matrix entry and its symmetric opposite twin
                Tvalue_a = self.get_fluid_diffusion_coefficient() * (self.get_G(gRow) + self.get_wavevector_q()) 
                Tvalue_b = self.get_theta_Pg(gRow, gCol) * (self.get_G(gCol) + self.get_wavevector_q())
                Tvalue = np.dot(Tvalue_a, Tvalue_b)
                self.set_Tgg(Tvalue, gRow, gCol)
                self.set_Tgg(Tvalue, gCol, gRow)
        return
    
    def build_matrix_Wgg(self):
        # - Reinitialize matrix
        self.Wgg = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        
        # - Build Wgg matrix 
        rows = self.get_reciprocal_points()
        cols = rows
        for gRow in range(rows):
            if(self._verbose):
                print("Wgg matrix:: building row", gRow, "out of", rows)
            for gCol in range(gRow, cols):
                # - Wgg matrix entry and its symmetric opposite twin
                Wvalue = (-1) * self.get_omega() * self.get_theta_Mg(gRow, gCol)
                if(gRow == gCol):
                    Wvalue += 1.0
                self.set_Wgg(Wvalue, gRow, gCol)
                self.set_Wgg(Wvalue, gCol, gRow)
        return    

    def build_matrix_invW(self):
        # - Reinitialize matrix
        self.invW = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))
        
        self.set_invW(np.linalg.inv(self.get_Wgg()))
        return
    
    def build_matrix_invWxT(self):
        # - Reinitialize matrix
        self.invWxT = np.asmatrix(np.zeros([self.get_reciprocal_points(), self.get_reciprocal_points()]))        
        
        self.set_invWxT(self.get_invW() * self.get_Tgg())
        return
    
    def solve_eigenvalues(self):
        # Reinitialize array and matrix
        self.eigen_values = np.zeros(self.get_reciprocal_points(), dtype=complex)
        self.eigen_vectors = np.asmatrix(np.zeros([self.get_reciprocal_points(),self.get_reciprocal_points()], dtype=complex)) 
        
        # Solve eigenvalue problem numerically
        eig = np.linalg.eig(self.get_invWxT())
        vals = eig[0]
        vecs = eig[1].transpose()

        # Sort eigen values and its vector
        inds = np.argsort(vals)
        vals = vals[inds]
        vecs = vecs[inds]       

        # Assign results to class members
        self.set_eigenvalues(vals)
        self.set_eigenvectors(vecs)        
        return
    
    def build_eigenstates(self):
        # Reinitialize array
        self.eigen_states = np.zeros(self.get_reciprocal_points(), dtype=complex)

        row = self.get_gk_index()
        print(row)
        indexes = self.get_reciprocal_points()
        cols = self.get_reciprocal_points()
        for n in range(indexes):
            eigenstate = 0.0
            for col in range(cols):
                eigenstate += self.get_theta_Pg(row, col) * self.get_eigenvector_coord(n, col) 
            
            self.set_eigenstate(n, eigenstate)
        return
    
    def build_Mkt(self):
        self.Mkt = np.zeros(self.get_k_points(), dtype=complex)
        return
    
    def solve_Mkt(self, _kIdx, _time):
        points = self.get_reciprocal_points()
        signal = 0.0
        for n in range(points):
            signal += np.exp((-1) * np.real(self.get_eigenvalue(n)) * _time) * ( np.abs(np.real(self.get_eigenstate(n)))**2 )

        self.set_Mkt(_kIdx, signal/self.get_porosity())     
        return
    
    def normalize_Mkt(self):
        M0t = self.get_Mkt(0)
        for idx in range(self.get_k_points()):
            normalized_Mkt = self.get_Mkt(idx) / M0t
            self.set_Mkt(idx, normalized_Mkt)
        
        return


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
        self._n_linspace_length = int(1 + (2 * self.get_unit_cell_divisions()))
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
    
    def set_porosity(self):
        x = self.get_grain_radius() / self.get_unit_cell_length()
        if (x > 0.5):
            self._porosity = 1.0 + (np.pi / 4.0) - (3.0 * np.pi * x**2) + ((8.0 / 3.0 ) * np.pi) * x**3 
        else:
            self._porosity = 1.0 - ((4.0 / 3.0 ) * np.pi) * x**3
        return
    
    def get_porosity(self):
        return self._porosity

    def set_n_points(self):
        self._n_points = self.get_n_linspace_length()**3
        return

    def get_n_points(self):
        return self._n_points
    
    def set_spatial_divisions(self, _value):
        self.spatial_divisions = _value
        self.set_essentials(False)
        return
    
    def get_spatial_divisions(self):
        return self.spatial_divisions
    
    def set_spatial_points(self):
        self.spatial_points = self.get_spatial_divisions()**3
        return
    
    def get_spatial_points(self):
        return self.spatial_points
    
    def set_reciprocal_divisions(self, _value):
        self.reciprocal_divisions = _value
        self.set_essentials(False)
        return
    
    def get_reciprocal_divisions(self):
        return self.reciprocal_divisions
    
    def set_reciprocal_points(self):
        self.reciprocal_points = (2*self.get_reciprocal_divisions() + 1)**3
        return
    
    def get_reciprocal_points(self):
        return self.reciprocal_points

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
    
    def set_theta_Mg(self, row, col, value):
        self.theta_Mg[row, col] = value
        return
    
    def get_theta_Mg(self, row=None, col=None):
        if(row == None):
            return self.theta_Mg
        else:
            return self.theta_Mg[row, col]
    
    def set_theta_Pg(self, row, col, value):
        self.theta_Pg[row, col] = value
        return
    
    def get_theta_Pg(self, row=None, col=None):
        if(row == None):
            return self.theta_Pg
        else:
            return self.theta_Pg[row, col]

    def set_k_points(self, _kpoints):
        self.k_points = _kpoints
        return
    
    def get_k_points(self):
        return self.k_points
    
    def set_ka_max(self, _kamax):
        self.ka_max = _kamax
        return
    
    def get_ka_max(self):
        return self.ka_max
    
    def set_wavevector_k(self, _k):
        self.wavevector_k = np.zeros([self.get_k_points(), 3])    
        kmax = (self.get_ka_max() / self.get_unit_cell_length()) * _k
        
        dKx = kmax[0] / (self.get_k_points() - 1)
        dKy = kmax[1] / (self.get_k_points() - 1)
        dKz = kmax[2] / (self.get_k_points() - 1) 
        for kpoint in range(self.get_k_points()):
            self.wavevector_k[kpoint][0] = kpoint * dKx
            self.wavevector_k[kpoint][1] = kpoint * dKy
            self.wavevector_k[kpoint][2] = kpoint * dKz 
        return
    
    def get_wavevector_k(self, _index=None):
        if (_index == None):
            return self.wavevector_k
        else:
            return self.wavevector_k[_index]
    
    def set_k_index(self, _index):
        self.k_index = _index
        return

    def get_k_index(self):
        return self.k_index
    
    # 'gk' is the reciprocal lattice vector g that is closest to k
    def set_wavevector_gk(self):
        gk_index = 0
        gk_value = self.get_G(gk_index)
        vecK = self.get_wavevector_k(self.get_k_index())
        minor_distance = np.linalg.norm(gk_value - vecK)

        for idx in range(1, self.get_reciprocal_points()):
            gk_value = self.get_G(idx)
            new_distance = np.linalg.norm(gk_value - vecK)
            if(new_distance < minor_distance):
                gk_index = idx
                minor_distance = new_distance
        
        self.set_gk_index(gk_index)
        self.wavevector_gk = self.get_G(gk_index)
        if(self._verbose):
            print("gk = {}, \tidx = {}, \tdistance = {}".format(self.get_wavevector_gk(), self.get_gk_index(), minor_distance))
        return
    
    def get_wavevector_gk(self):
        return self.wavevector_gk
    
    def set_gk_index(self, _index):
        self.gk_index = _index
        return
    
    def get_gk_index(self):
        return self.gk_index
    
    def set_wavevector_q(self):
        self.wavevector_q = self.get_wavevector_k(self.get_k_index()) - self.get_wavevector_gk()
        return
    
    def get_wavevector_q(self):
        return self.wavevector_q
    
    def set_Tgg(self, value, row, column):
        self.Tgg[row, column] = value
        return
    
    def get_Tgg(self, row=None, column=None):
        if(row == None):
            return self.Tgg
        else:
            return self.Tgg[row, column]
    
    def set_Wgg(self, value, row, column):
        self.Wgg[row, column] = value
        return
    
    def get_Wgg(self, row=None, column=None):
        if(row == None):
            return self.Wgg
        else:
            return self.Wgg[row, column]
    
    def set_invW(self, _invW):
        self.invW = _invW
        return
    
    def get_invW(self, row=None, column=None):
        if(row == None):
            return self.invW
        else:
            return self.invW[row, column]
    
    def set_invWxT(self, _invWxt):
        self.invWxT = _invWxt
        return
    
    def get_invWxT(self, row=None, column=None):
        if(row == None):
            return self.invWxT
        else:
            return self.invWxT[row, column]
    
    def set_eigenvalues(self, eigenvalues):
        self.eigen_values = eigenvalues
        return
    
    def get_eigenvalues(self):
        return self.eigen_values
    
    def get_eigenvalue(self, index):
        return self.eigen_values[index]
    
    def set_eigenvectors(self, eigenvectors):
        self.eigen_vectors = eigenvectors
        return
    
    def get_eigenvectors(self):
        return self.eigen_vectors
    
    def get_eigenvector(self, index):
        return self.eigen_vectors[index]
    
    def get_eigenvector_coord(self, index, coord):
        return self.eigen_vectors[index, coord]
    
    def set_eigenstate(self, _index, _eigenstate):
        self.eigen_states[_index] = _eigenstate
        return
    
    def get_eigenstates(self):
        return self.eigen_states
    
    def get_eigenstate(self, index):
        return self.eigen_states[index]
    
    def set_Mkt(self, _index, _value):
        self.Mkt[_index] = _value
        return
    
    def get_Mkt(self, _index=None):
        if (_index == None):
            return self.Mkt
        else:
            return self.Mkt[_index]
    
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
        n_points = self.get_spatial_points()
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

    def dataviz_Mkt(self):
        # Points (s=point_size, c=color, cmap=colormap)
        n_points = self.get_k_points()
        
        data_x = [] 
        for idx in range(n_points):
            data_x.append(np.linalg.norm(self.get_wavevector_k(idx)) * self.get_unit_cell_length()) 
        
        data_y = [] 
        for idx in range(n_points):
            data_y.append(np.real(self.get_Mkt(idx)))       
                
        plt.semilogy(data_x, data_y, '-o')
        plt.title("")
        plt.xlabel(r'$ |k|a $')
        plt.ylabel(r'$ M(k,t) $')

        # Show the major grid lines with dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

        # Set plot axes limits
        plt.xlim(0.0, 1.05*data_x[-1])
        # plt.ylim()
        plt.show()
        return

    
