#ifndef __MATRIX_BASE_H
#define __MATRIX_BASE_H
#include <iostream>
#include <string>
#include <memory>
#include <cassert>
#include <complex>
#include <algorithm>
#include <utility>
#include "/home/peter/UTILS/LAPACK-3.8.0/LAPACKE/include/lapacke.h"


template<typename DataType> 
class Matrix_Base  {
  protected :
    int nrows_;
    int ncols_;
    int size_;
   
  public : 

    int nrows() const { return nrows_; }
    int ncols() const { return ncols_; }
    int size() const { return size_; }

    virtual DataType element( int ii, int jj ) { return (DataType)(0.0);}

    virtual void scale( double factor){};
    virtual void scale( std::complex<double> factor ){};

    virtual void transpose(){};
    virtual void hconj(){};

    virtual void print(){};

    Matrix_Base(){};
    Matrix_Base( int nrows, int ncols ) : nrows_(nrows), ncols_(ncols), size_(nrows*ncols) {};

    ~Matrix_Base(){};

}; 
#endif
