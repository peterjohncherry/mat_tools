#include "vector_base.h" 


template<>
Vector_Base<double>::Vector_Base( const int& size ) : size_(size){} 
 
template<>
Vector_Base<std::complex<double>>::Vector_Base( const int& size ) : size_(size){} 
  
template class Vector_Base<double>;
template class Vector_Base<std::complex<double>>;
