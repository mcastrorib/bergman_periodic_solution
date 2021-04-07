#include <iostream>
#include <Eigen/Dense>
#include <cmath>
 
using namespace std;
using namespace Eigen;

double *** alloc3DArray(int dimX, int dimY, int dimZ);
void print3DArray(double ***array, int dimX, int dimY, int dimZ);
void free3DArray(double ***array, int dimX, int dimY, int dimZ);
double setPorosity(double _a, double _r);

int main()
{
    int N = 2;
    double Dp = 2.5;
    double Dm = 0.0;
    double cellLength = 10.0;
    double sphereRadius = 5.0;
    double w = 0.999999;
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
    
    cout << "times: \n" << times_Array << endl;
    cout << "M(k,t): \n" << M_kt << endl;

    // start computation
    double volume = pow(cellLength, 3);
    double porosity = setPorosity(cellLength, sphereRadius);
    int points = 2*N + 1;
    int points3D = points*points*points;

    // create spatial arrays
    ArrayXd rawX = ArrayXd::LinSpaced(2*points + 1, -0.5*cellLength, 0.5*cellLength);
    ArrayXd vecX(points); ArrayXd vecY(points); ArrayXd vecZ(points);
    for(int i = 0; i < points; i++)
    {
        vecX(i) = rawX(1 + 2*i);
        vecY(i) = rawX(1 + 2*i);
        vecZ(i) = rawX(1 + 2*i);
    }
    
    // create wavevector arrays
    double Gfreq = 2.0 * M_PI / cellLength; 
    ArrayXd vecGX = Gfreq * ArrayXd::LinSpaced(points, -((double) N), ((double) N)); 
    ArrayXd vecGY = Gfreq * ArrayXd::LinSpaced(points, -((double) N), ((double) N)); 
    ArrayXd vecGZ = Gfreq * ArrayXd::LinSpaced(points, -((double) N), ((double) N));

    cout << vecGX << endl;

    // create 3D array
    ArrayXXd poreGrid(3, points3D);
    for (int k = 0; k < points; k++)
    {
        for (int i = 0; i < points; i++)
        {
            for (int j = 0; j < points; j++)
            {
                int index = j + i*points + k*points*points;
                cout << index << endl;
                poreGrid(0, index) = vecX(j);
                poreGrid(1, index) = vecY(i);
                poreGrid(2, index) = vecZ(k);                 
            }   
        }   
    }

    for(int i = 0; i < points3D; i++)
        cout << "pore grid: \n" << poreGrid.col(i) << endl;
}

double setPorosity(double _a, double _r)
{
    double x = _r/_a;
	if(x <= 0.5)
		return 1.0 - ((4.0/3.0) * M_PI * (pow(x,3)));
	else
		return 1.0 + (1.0/4.0)*M_PI - 3.0*M_PI*x*x + (8.0/3.0)*M_PI*pow(x,3);
}

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
                cout << "[" << i;
                cout << "][" << j;
                cout << "][" << k;
                cout << "] = " << array[i][j][k] << ", ";                 
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