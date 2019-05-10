#ifndef __ZVECTOR_H
#define __ZVECTOR_H
#include "rvector.h"

class ZVector : public Vector_Base<std::complex<double>>  {
 
  private : 
    std::unique_ptr<RVector> real_vec_;
    std::unique_ptr<RVector> imag_vec_;
 
  public : 

    std::complex<double> dot_product( const ZVector& vec ) const;
    inline std::complex<double> dot( const ZVector& vec) const { return dot_product(vec);};

    inline std::complex<double> element(const int& ii) const {
       return std::complex<double>( real_vec_->element(ii), imag_vec_->element(ii) );
    }

    void print();
    void scale( const std::complex<double>& factor ) {};
    void scale( const double& factor ) {};

    ZVector() : Vector_Base<std::complex<double>>(){};
 
    ZVector( const ZVector& vec);
 
    ZVector(const int& size );
 
    ZVector(const int& size, const std::complex<double>& init_val );
 
    ZVector(const int& size, const std::unique_ptr<double[]>& init_data );
 
    ZVector( const int& size, const std::unique_ptr<double[]>& init_real_data, 
                              const std::unique_ptr<double[]>& init_imag_data );
 
    ZVector( const int& size, const double* init_real_data, const double* init_imag_data );
  
    ~ZVector(){};

};
#endif
