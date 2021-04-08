#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
 
using namespace std;
using namespace Eigen;

double *** alloc3DArray(int dimX, int dimY, int dimZ);
void print3DArray(double ***array, int dimX, int dimY, int dimZ);
void free3DArray(double ***array, int dimX, int dimY, int dimZ);
double setPorosity(double _a, double _r);
int IDX2C_3D(int i, int j, int k, int dim);
RowVector3i revertIDX2C_3D(int i, int dim);
int mapGTodG(int idx1, int idx2, int dim);
ArrayXXd createGrid3d(ArrayXd& points, int dim);
int checkPorePhase(double x, double y, double z, double radius);
int checkPorePhase(Vector3d position, double radius);
void printPhase(ArrayXd& phase, int dim, string phaseName);
void printPhase(ArrayXcd& phase, int dim, string phaseName);
void apply_dft(ArrayXcd& phase_kX, ArrayXd& phase_X, ArrayXXd& grid_X, ArrayXXd& grid_G, double volume, int dim);
void apply_symmetric_dft_identity(ArrayXcd& matrix_kX, ArrayXcd& phase_kX, int dim);
int findGkIndex(Vector3cd &kVec, ArrayXXd &gridG, int dim);

int main()
{
    int N = 1;
    double Dp = 2.5;
    double Dm = 0.0;
    double cellLength = 10.0;
    double sphereRadius = 5.0;
    double w = 0.9999;
    double u = 1.0;
    double rho = 0.0;
    double spuriousCut = 0.25;

    // create array with wave vector k
    int ka_Points = 40;
    Vector3d ka_Direction(1.0, 0.0, 0.0);
    double ka_Min = 0.01;
    double ka_Max = 5.0 * M_PI;
    ArrayXXd ka_Table(3, ka_Points);
    for(int coord = 0; coord < 3; coord++) 
        ka_Table.row(coord) = ka_Direction(coord) * ArrayXd::LinSpaced(ka_Points, ka_Min / cellLength, ka_Max / cellLength);


    // create time samples
    int time_Samples = 8;
    double time_Scale = cellLength*cellLength/Dp;
    ArrayXd times_Array(time_Samples);
    times_Array << 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0, 100.0;


    // create array to store results
    ArrayXXd M_kt = ArrayXXd::Zero(time_Samples, ka_Points);
    
    // start computation
    double volume = pow(cellLength, 3);
    double porosity = setPorosity(cellLength, sphereRadius);
    int points = 2*N + 1;
    int points3D = points*points*points;

    // create spatial arrays
    ArrayXd raw_R = ArrayXd::LinSpaced(2*points + 1, -0.5*cellLength, 0.5*cellLength);
    ArrayXd vec_R(points);
    for(int i = 0; i < points; i++)
    {
        vec_R(i) = raw_R(1 + 2*i);
    }

    // create wavevector arrays
    double Gfreq = 2.0 * M_PI / cellLength; 
    ArrayXd vec_G = Gfreq * ArrayXd::LinSpaced(points, -((double) N), ((double) N)); 
    
    // create 3D array (spatial and wavevector grids)
    ArrayXXd grid_R = createGrid3d(vec_R, points);
    ArrayXXd grid_G = createGrid3d(vec_G, points);
    
    // start differential space
    int dpoints = 4*N + 1;
    int dpoints3D = dpoints*dpoints*dpoints;

    // create spatial arrays
    ArrayXd raw_dR = ArrayXd::LinSpaced(2*dpoints + 1, -0.5*cellLength, 0.5*cellLength);
    ArrayXd vec_dR(dpoints);
    for(int i = 0; i < dpoints; i++)
    {
        vec_dR(i) = raw_dR(1 + 2*i);
    }
    ArrayXXd grid_dR = createGrid3d(vec_dR, dpoints);

    // create wavevector arrays
    ArrayXd vec_dG = Gfreq * ArrayXd::LinSpaced(dpoints, -((double) 2.0*N), ((double) 2.0*N)); 
    ArrayXXd grid_dG = createGrid3d(vec_dG, dpoints);

    // create spatial pore and matrix phase arrays
    ArrayXd pore_dR(dpoints3D);
    for (int k = 0; k < dpoints; k++)
    {
        for (int i = 0; i < dpoints; i++)
        {
            for (int j = 0; j < dpoints; j++)
            {
                int index = IDX2C_3D(i,j,k,dpoints);
                Vector3d position(grid_dR(0, index), grid_dR(1, index), grid_dR(2, index));
                pore_dR(index) = checkPorePhase(position, sphereRadius);                 
            }   
        }   
    }

    ArrayXd matrix_dR(dpoints3D);
    for (int index = 0; index < dpoints3D; index++)
    {
        if(pore_dR(index) == 1) matrix_dR(index) = 0;
        else matrix_dR(index) = 1;
    }

    // create wavenumber pore and matrix phase arrays
    ArrayXcd pore_dG(dpoints3D); apply_dft(pore_dG, pore_dR, grid_dR, grid_dG, volume, dpoints);
    ArrayXcd matrix_dG(dpoints3D); apply_symmetric_dft_identity(matrix_dG, pore_dG, dpoints); 

    /*
        Here the actual complex matrix computations start 
    */ 
    int mRows = pow(points, 3);
    int mCols = mRows;
    MatrixXcd matW = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matU = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matR = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matRinv = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matRH = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matRHinv = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matRRH = MatrixXcd::Zero(mRows, mCols);
    MatrixXcd matT = MatrixXcd::Zero(mRows, mCols);

    // MatW assembly
    double ***occurs = alloc3DArray(dpoints, dpoints, dpoints);
    cout << endl << "** ASSEMBLYING MATRIX W **" << endl;
    for(int k = 0; k < points; k++) 
    {
        for(int i = 0; i < points; i++)
        {
            for(int j = 0; j < points; j++)
            {
                int row_index = IDX2C_3D(i, j, k, points);
                cout << ":: Matrix_W row " << row_index << " out of " << mRows << endl;

                for(int kk = 0; kk < points; kk++) 
                {
                    for(int ii = 0; ii < points; ii++) 
                    {
                        for(int jj = 0; jj < points; jj++)
                        {
                            int col_index = IDX2C_3D(ii, jj, kk, points);

                            int di = mapGTodG(i, ii, N);
                            int dj = mapGTodG(j, jj, N);
                            int dk = mapGTodG(k, kk, N);
                            int dIndex = IDX2C_3D(di, dj, dk, dpoints);
                            occurs[di][dj][dk] += 1.0;
                            matW(row_index, col_index) = (-1.0) * w * matrix_dG(dIndex);  
                        }
                    }
                }
            }
        }    
    }
    for(int row = 0; row < mRows; row++) 
        matW(row, row) += 1.0;

    // Get cholesky decomposition of matrix R and agreggates
    cout << endl << "** APPLYING CHOLESKY DECOMPOSITION ON MATRIX W **" << endl;
    matRH = matW.llt().matrixL();
    matRHinv = matRH.inverse();
    matR = matRH.adjoint();
    matRinv = matR.inverse();
    matRRH = matR * matRH;

    cout << endl << "** ASSEMBLYING MATRIX U **" << endl;
    for(int k = 0; k < points; k++) 
    {
        for(int i = 0; i < points; i++)
        {
            for(int j = 0; j < points; j++)
            {
                int row_index = IDX2C_3D(i, j, k, points);
                cout << ":: Matrix_U row " << row_index << " out of " << mRows << endl;

                for(int kk = 0; kk < points; kk++) 
                {
                    for(int ii = 0; ii < points; ii++) 
                    {
                        for(int jj = 0; jj < points; jj++)
                        {
                            int col_index = IDX2C_3D(ii, jj, kk, points);

                            int di = mapGTodG(i, ii, N);
                            int dj = mapGTodG(j, jj, N);
                            int dk = mapGTodG(k, kk, N);
                            int dIndex = IDX2C_3D(di, dj, dk, dpoints);
                            matU(row_index, col_index) = (-1.0) * u * matrix_dG(dIndex);  
                        }
                    }
                }
            }
        }    
    }
    for(int row = 0; row < mRows; row++) 
        matU(row, row) += 1.0;

    // Solve eigenvalue problem for q
    cout << endl << "** SOLVING EIGENVALUE PROBLEM FOR k **" << endl;
    ArrayXXcd vals_q = ArrayXXcd::Zero(mRows, ka_Points);
    ArrayXXcd weights_q = ArrayXXcd::Zero(mRows, ka_Points);
    ArrayXXcd spurs_q = ArrayXXcd::Zero(mRows, ka_Points);
    for(int kIndex = 0; kIndex < ka_Points; kIndex++)
    {
        Vector3cd kVec(ka_Table(0,kIndex), ka_Table(1,kIndex), ka_Table(2,kIndex));
        int GkIndex = findGkIndex(kVec, grid_G, points);
        Vector3cd GkVec(grid_G(0, GkIndex), grid_G(1, GkIndex), grid_G(2, GkIndex));
        Vector3cd qVec = kVec - GkVec;
        
        cout << "---------------------------------" << endl;
        cout << "kIndex = \t" << kIndex << "\t GkIndex = \t" << GkIndex << endl;
        cout << "k = \t" << kVec.transpose() << endl;
        cout << "gk = \t" << GkVec.transpose() << endl;
        cout << "q = \t" << qVec.transpose() << endl;      
        cout << "---------------------------------" << endl << endl;

    }

    // debug region
    // for(int row = 0; row < mRows; row++)
    //     cout << "matW row "<< row << "\n" << matW.row(row) << endl;
    

    return 0;
}

int findGkIndex(Vector3cd &kVec, ArrayXXd &gridG, int dim)
{
    int half = dim/2;
    int GkIndex = IDX2C_3D(half, half, half, dim);
    Vector3cd GkVec(gridG(0, GkIndex), gridG(1, GkIndex), gridG(2, GkIndex));
    Vector3cd diffVec = kVec - GkVec;
    double distance = diffVec.norm();

    // Set direction X search space
    int kXmin = half;
    int kXmax = dim;
    int kXinc = 1;
    if(abs(kVec(0)) < 0.0)
    {
        kXmax = -1;
        kXinc = -1;
    }

    // Set direction Y search space
    int kYmin = half;
    int kYmax = dim;
    int kYinc = 1;
    if(abs(kVec(1)) < 0.0)
    {
        kYmax = -1;
        kYinc = -1;
    }

    // Set direction Y search space
    int kZmin = half;
    int kZmax = dim;
    int kZinc = 1;
    if(abs(kVec(2)) < 0.0)
    {
        kZmax = -1;
        kZinc = -1;
    }

    for(int k = kZmin; k != kZmax; k += kZinc)
    {
        for(int i = kYmin; i != kYmax; i += kYinc)
        {
            for(int j = kXmin; j != kXmax; j += kXinc)
            {
                int gIndex = IDX2C_3D(i,j,k,dim);
                Vector3cd gVec(gridG(0, gIndex), gridG(1, gIndex), gridG(2, gIndex));
                diffVec = kVec - gVec;
                double newDistance = diffVec.norm();

                if(newDistance < distance)
                {
                    distance = newDistance;
                    GkIndex = gIndex;
                }
            }
        }        
    }

    return GkIndex;
}

ArrayXXd createGrid3d(ArrayXd& points, int dim)
{
    ArrayXXd grid(3, dim*dim*dim);
    for (int k = 0; k < dim; k++)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                int index = IDX2C_3D(i,j,k,dim);
                grid(0, index) = points(j);
                grid(1, index) = points(i);
                grid(2, index) = points(k);                 
            }   
        }   
    }

    return grid;
}

int checkPorePhase(double x, double y, double z, double radius)
{
    Vector3d point(x,y,z);
    if(point.norm() > radius)
        return 1;
    else
        return 0;
}

int checkPorePhase(Vector3d position, double radius)
{
    if(position.norm() > radius)
        return 1;
    else
        return 0;
}

void printPhase(ArrayXd& phase, int dim, string phaseName)
{
    cout << endl << phaseName << " phase:" << endl;
    for(int slice = 0; slice < dim; slice++)
    {
        cout << endl << "z = " << slice << ":" << endl;
        for (int row = 0; row < dim; row++)
        {
            for (int col = 0; col < dim; col++)
            {
                int index = IDX2C_3D(row, col, slice, dim);
                cout << phase(index) << " ";
            }
            cout << endl;
        }
    }
}

void printPhase(ArrayXcd& phase, int dim, string phaseName)
{
    cout << endl << phaseName << " phase:" << endl;
    for(int slice = 0; slice < dim; slice++)
    {
        cout << endl << "z = " << slice << ":" << endl;
        for (int row = 0; row < dim; row++)
        {
            for (int col = 0; col < dim; col++)
            {
                int index = IDX2C_3D(row, col, slice, dim);
                cout << phase(index) << " ";
            }
            cout << endl;
        }
    }
}

int IDX2C_3D(int i, int j, int k, int dim)
{
    return (j + i*dim + k*dim*dim);
}

RowVector3i revertIDX2C_3D(int i, int dim)
{
    RowVector3i vec((i/dim % dim), (i%dim), (i/(dim*dim)));
    return vec;
}

int mapGTodG(int idx1, int idx2, int dim)
{
    return (idx1 - idx2 + 2*dim);
}

double setPorosity(double _a, double _r)
{
    double x = _r/_a;
	if(x <= 0.5)
		return 1.0 - ((4.0/3.0) * M_PI * (pow(x,3)));
	else
		return 1.0 + (1.0/4.0)*M_PI - 3.0*M_PI*x*x + (8.0/3.0)*M_PI*pow(x,3);
}

void apply_dft(ArrayXcd& phase_kX, ArrayXd& phase_R, ArrayXXd& grid_R, ArrayXXd& grid_G, double volume, int dim)
{
    cout << endl << "** APPLYING DFT **" << endl;
    int elems = pow(dim,3);
    double dV = volume / (double) elems;
    int count = 0;   
    int sliceDim = dim*dim;
    complex <double> minusJ(0,-1);
    for(int k = 0; k < dim; k++) 
    {
        for(int i = 0; i < dim; i++) 
        {
            for(int j = 0; j < dim; j++) 
            {
                count += 1;
                if(count % sliceDim == 0) cout << ":: Pore_dg " << count << " out of " << elems << endl;
                int index = IDX2C_3D(i,j,k,dim);
                Vector3cd dG(grid_G(0,index), grid_G(1,index), grid_G(2,index));
                complex<double> gSum = 0.0;
                for(int rz = 0; rz < dim; rz++)
                {
                    for(int ry = 0; ry < dim; ry++)
                    {
                        for(int rx = 0; rx < dim; rx++)
                        {   
                            int rIndex = IDX2C_3D(rx, ry, rz, dim);
                            Vector3cd dR(grid_R(0,rIndex), grid_R(1,rIndex), grid_R(2,rIndex));
                            complex<double> firstTerm = dV * phase_R(rIndex);
                            gSum += firstTerm * exp(minusJ * dG.dot(dR));
                        }
                    }
                }     

                phase_kX(index) = (1.0/volume) * gSum;
            }
        }
    }
}

void apply_symmetric_dft_identity(ArrayXcd& matrix_kX, ArrayXcd& pore_kX, int dim)
{
    cout << endl << "** APPLYING DFT SYMMETRIC IDENTITY **" << endl;
    int count = 0;
    int elems = pow(dim,3);
    int sliceDim = dim*dim;
    for(int k = 0; k < dim; k++)
    {
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                count += 1;
                if(count % sliceDim == 0) cout << ":: Matrix_dg " << count << " out of " << elems << endl;
                int index = IDX2C_3D(i,j,k,dim);
                matrix_kX(index) = (-1.0) * pore_kX(index);
            }
        }
    }

    // increment in diagonal terms
    complex<double> unity(1.0, 0.0);
    int dIndex = IDX2C_3D(dim/2, dim/2, dim/2, dim);
    matrix_kX(dIndex) += unity;
}

// pointer*** functions
double *** alloc3DArray(int dimX, int dimY, int dimZ)
{
    double ***array = new double**[dimZ];
    for(int i = 0; i < dimY; i++)
    {
        array[i] = new double*[dimY];
        for(int j = 0; j < dimX; j++)
        {
            array[i][j] = new double[dimX];
        }
    }

    return array;
}


void print3DArray(double ***array, int dimX, int dimY, int dimZ)
{
    for (int k = 0; k < dimZ; k++)
    {
        cout << "slice " << k << endl;
        for (int i = 0; i < dimY; i++)
        {
            for (int j = 0; j < dimX; j++)
            {
                cout << array[i][j][k] << "\t";                 
            }   
            cout << endl;
        }   
    }
}

void free3DArray(double ***array, int dimX, int dimY, int dimZ)
{
    for(int i = 0; i < dimY; i++)
    {
        for(int j = 0; j < dimX; j++)
        {
            delete [] array[i][j];
            array[i][j] = NULL;
        }
        delete [] array[i];
        array[i] = NULL;
    }

    delete [] array;
    array = NULL;
}