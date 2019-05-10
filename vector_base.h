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
   
  public : 
  
    int size() const { return size_; }

    virtual DataType element( const int& ii ) const = 0;
    virtual double norm() const = 0;
    virtual void scale( const DataType& factor) = 0;
    virtual void print() const = 0;
    

    Vector_Base(){};
    Vector_Base( const int& size ) : size_(size) {};

    ~Vector_Base(){};

}; 
#endif
