// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <limits>

// include OpenMP for multicore implementation
#include <omp.h>

// Class header
#include "LeastSquareAdjust.h"

LeastSquareAdjust::LeastSquareAdjust(VectorXd &_x, 
									 VectorXd &_y, 
									 bool _verbose): X(_x), 
									 				 Y(_y),
													 verbose(_verbose), 
													 meanX(0.0),
													 meanY(0.0), 
													 A(0.0), 
													 B(0.0), 
													 solved(false)
{
	this->begin = 0;
	this->end = this->X.size();
	this->points = this->end + 1;
	this->threshold = numeric_limits<double>::max();
}

void LeastSquareAdjust::setX(VectorXd &_x)
{
	this->X = _x;
	(*this).setAsUnsolved();
}

void LeastSquareAdjust::setY(VectorXd &_y)
{
	this->Y = _y;
	(*this).setAsUnsolved();
}

void LeastSquareAdjust::setThreshold(double _threshold)
{
	this->threshold = _threshold;
	// cout << "threshold: " << this->threshold << endl;
	(*this).setLimits();
}

void LeastSquareAdjust::setPoints(int _points)
{
	this->points = _points;
	if(_points > 0 and _points < this->end)
	{
		(*this).setThreshold(this->X(_points - 1));
	}
}

void LeastSquareAdjust::setLimits()
{
	int idx = this->begin;
	bool limitExceeded = false;

	while(idx < this->end && limitExceeded == false)
	{
		if(this->threshold <= fabs(this->X(idx))) 
			limitExceeded = true;
		
		idx++;
	}

	if(limitExceeded) this->end = idx;
}

void LeastSquareAdjust::solve()
{
	this->meanX = computeMean(this->X);
	this->meanY = computeMean(this->Y);
	(*this).computeB();
	(*this).computeA();
	(*this).setAsSolved();
	if(this->verbose)
		cout << "A: " << this->A << "\t" << "B: " << this->B << endl;
}    

double LeastSquareAdjust::computeMean(VectorXd &_vector)
{
	// double sum = 0.0;
	// double size = (double) _vector.size();
	// for(uint idx = this->begin; idx < this->end; idx++)
	// {
	// 	sum += _vector(idx);
	// }

	// return (sum/size);
	
	return _vector.mean();
}

void LeastSquareAdjust::computeB()
{
	if(this->verbose) 
	{
		cout << "samples adjusted by least-squares approach: [";
		cout << this->begin << " to " << this->end - 1 << "]" << endl;
	}
	
	// get B dividend
	double dividend = 0.0;
	for(uint idx = this->begin; idx < this->end; idx++)
	{
		dividend += this->X(idx) * (this->Y(idx) - this->meanY);
	}

	// get B divisor
	double divisor = 0.0;
	for(uint idx = this->begin; idx < this->end; idx++)
	{
		divisor += this->X(idx) * (this->X(idx) - this->meanX);
	}

	this->B = (dividend/divisor);
}

void LeastSquareAdjust::computeA()
{
	this->A = this->meanY - (this->B * this->meanX);
}

void LeastSquareAdjust::setAsSolved()
{
	this->solved = true;
}

void LeastSquareAdjust::setAsUnsolved()
{
	this->solved = false;
}
