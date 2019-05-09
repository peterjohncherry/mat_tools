#ifndef __VECTOR_BASE_H
#define __VECTOR_BASE_H
#include <iostream>
#include <string>
#include <memory>
#include <cassert>
#include <complex>
#include <algorithm>
#include <utility>


template<typename DataType> 
class Vector_Base  {
  protected :

    int size_;
    std::unique_ptr<double[]> data_;
    double* data_ptr_;
   
  public : 
  
    int size() const { return size_; }
    double* data_ptr() const { return data_ptr_; }
    double* element_ptr(const int& ii) const { return data_ptr_+ii; }

    virtual DataType element( const int& ii ) const = 0;
    virtual void scale( DataType factor) = 0;
    virtual void print() = 0;
    

    Vector_Base(){};
    Vector_Base( const int& size ) : size_(size) {};
    Vector_Base( const int& size, std::unique_ptr<double[]>& );

    ~Vector_Base(){};

}; 
#endif
