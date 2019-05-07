#ifndef __RVECTOR_H
#define __RVECTOR_H
#include "vector_base.h"

class RVector : public Vector_Base<double>  {
  
  private : 
    std::unique_ptr<double[]> data_;
    double* data_ptr_;
   
  public : 
    double* data_ptr() const { return data_ptr_; }
    double  element(const int& ii) const { return *(data_ptr_+ii); }
    double* element_ptr(const int& ii) const { return data_ptr_+ii; }

    double dot_product( const RVector& vec) const;

    void print();

    RVector() : Vector_Base<double>(){};
    RVector( RVector& vec);
    RVector(const int& size );
    RVector(const int& size, const double& init_val );
    RVector(const int& size, const std::unique_ptr<double[]>& init_data );
  
    ~RVector(){};

};
#endif
