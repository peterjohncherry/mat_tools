#ifndef __ZMATRIX_H
#define __ZMATRIX_H
#include "rmatrix.h"

class ZMatrix : public Matrix_Base<std::complex<double>>   {
  private :
    std::unique_ptr<RMatrix> real_mat_;
    std::unique_ptr<RMatrix> imag_mat_;
    std::unique_ptr<double[]> complex_data_;
    std::unique_ptr<std::complex<double>[]> stdcomplex_data_;

    std::unique_ptr<ZMatrix> r_eigenvectors_;
    std::unique_ptr<ZMatrix> l_eigenvectors_;

  public : 
   
    void generate_complex_data();
    void generate_stdcomplex_data();

    double* real_data_ptr() const { return real_mat_->data_ptr(); }
    double* imag_data_ptr() const { return imag_mat_->data_ptr(); }
    double* complex_data_ptr() const { return complex_data_.get(); }
    std::complex<double>* stdcomplex_data_ptr() const { return stdcomplex_data_.get(); }

    std::complex<double> element(const int& ii, const int& jj) const {
      return std::complex<double>(real_mat_->element(ii,jj), imag_mat_->element( ii, jj) );
    }

    void set_test_elems();
    void transpose();
    void hconj();
    void scale( double factor );
    void scale( std::complex<double> factor );


    void diagonalize_complex_routine(std::unique_ptr<ZMatrix> mat );
    void diagonalize_stdcomplex_routine(std::unique_ptr<ZMatrix>& mat );
   

    std::unique_ptr<ZMatrix> ax_plus_b( const std::unique_ptr<ZMatrix>& matrix_b, double factor );
    std::unique_ptr<ZMatrix> ax_plus_b( const std::unique_ptr<ZMatrix>& matrix_b, std::complex<double> factor );

    std::unique_ptr<ZMatrix> multiply( std::unique_ptr<ZMatrix>& matb );
    void  diagonalize_stdcomplex_routine();

    void print();

    ZMatrix( int ncols, int nrows );
    ZMatrix( int ncols, int nrows, const std::complex<double>& init_val);
    ZMatrix( int ncols, int nrows, const std::unique_ptr<double[]>& init_data );
    ZMatrix( int ncols, int nrows, const std::unique_ptr<std::complex<double>[]>& init_data );
    ~ZMatrix(){};

};
#endif
