#ifndef LSADJUST_H
#define LSADJUST_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

using namespace std;
using namespace Eigen;

class LeastSquareAdjust
{
public:
    VectorXd &X;
    VectorXd &Y;    
    
    LeastSquareAdjust(VectorXd &_x, VectorXd &_y, bool _verbose=true);
    virtual ~LeastSquareAdjust(){}

    void setX(VectorXd &_x);
    void setY(VectorXd &_y);
    void setThreshold(double _threshold);
    void setPoints(int _points);
    void setLimits();

    void solve();    

    double getMeanX(){ return this->meanX; }
    double getMeanY(){ return this->meanY; }
    double getA(){ return this->A; }
    double getB(){ return this->B; }

private:
    bool verbose;
    double meanX, meanY;
    double A, B;
    bool solved;

    int begin, end;
    int points;
    double threshold;

    double computeMean(VectorXd &_vector);
    void computeB();
    void computeA();
    void setAsSolved();
    void setAsUnsolved();
};

#endif
