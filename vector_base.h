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

    virtual DataType element( const int& ii ) const { return (DataType)(0.0);}
    virtual DataType* element_ptr( const int& ii ) const = 0;

    virtual void scale( DataType factor){};

    virtual void print(){};

    Vector_Base(){};
    Vector_Base( const int& size ) : size_(size) {};
    Vector_Base( const int& size, std::unique_ptr<DataType[]>& ) : size_(size) {};

    ~Vector_Base(){};

}; 
#endif
