#ifndef __RMATRIX_H
#define __RMATRIX_H
#include "matrix_base.h"
#include "rvector.h"

class RMatrix : public Matrix_Base<double>  {
  
  private : 
    std::unique_ptr<RMatrix> r_eigenvectors_;
    std::unique_ptr<RMatrix> l_eigenvectors_;
    std::unique_ptr<double[]> eigenvalues_real_;
    std::unique_ptr<double[]> eigenvalues_imag_;
   
  public : 
    double* data_ptr_;
    double* data_ptr() const { return data_ptr_; }
    std::unique_ptr<double[]> data_;

    double  element(const int& ii, const int& jj) const { return *(data_ptr_ + jj*nrows_  + ii ); }
    double* element_ptr( const int& ii, const int& jj) const { return data_ptr_ + jj*nrows_ + ii; }

    void scale(double factor);
    void scale( std::complex<double> factor ){ throw std::logic_error("cannot scale rmatrix by complex factor"); }


    void transpose();
    void hconj(){ transpose(); }

    void set_test_elems();
    void symmetrize();

    std::unique_ptr<RMatrix> ax_plus_b( const std::unique_ptr<RMatrix>& matrix_b, double factor );
    std::unique_ptr<RMatrix> multiply( const std::unique_ptr<RMatrix>& matrix_b );
    std::unique_ptr<RVector> matvec_mult_lapack( std::unique_ptr<RVector>& vec );
    
    void diagonalize(); 

    void print();

    inline std::unique_ptr<RMatrix> operator + (const std::unique_ptr<RMatrix>& y)
    {
       return ax_plus_b(y,1.0);
    }

    inline std::unique_ptr<RMatrix> operator - (const std::unique_ptr<RMatrix>& y)
    {
       return ax_plus_b(y,-1.0);
    }

    inline std::unique_ptr<RMatrix> operator * (const std::unique_ptr<RMatrix>& BB)
    {
       return multiply(BB);
    }

    RMatrix() : Matrix_Base<double>(){};
    RMatrix( RMatrix& mat);
    RMatrix(int nrows, int ncols );
    RMatrix(int nrows, int ncols, const double& init_val );
    RMatrix(int nrows, int ncols, const std::unique_ptr<double[]>& init_data );
  
    ~RMatrix(){};

};
#endif
