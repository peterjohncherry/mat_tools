#ifndef __ZVECTOR_H
#define __ZVECTOR_H
#include "vector_base.h"

class ZVector : public Vector_Base<std::complex<double>>  {
  
  public : 

    std::complex<double> dot_product( const ZVector& vec ) const;
    inline std::complex<double> dot( const ZVector& vec) const { return dot_product(vec);};

    inline std::complex<double> element(int ii) const {
       return std::complex<double>( *(data_ptr_ +ii), *(data_ptr_ + size_/2 +ii) );
    }

    void print();

    ZVector() : Vector_Base<std::complex<double>>(){};
    ZVector( ZVector& vec);
    ZVector(const int& size );
    ZVector(const int& size, const std::complex<double>& init_val );
    ZVector(const int& size, const std::unique_ptr<double[]>& init_data );
  
    ~ZVector(){};

};
#endif
