#include "vector_base.h" 


template<>
Vector_Base<double>::Vector_Base( const int& size ) : size_(size){} 
 
template<>
Vector_Base<double>::Vector_Base( const int& size, std::unique_ptr<double[]>& init_data ) :
        size_(size), data_(std::make_unique<double[]>(size_)) {
   std::copy_n(init_data.get(), size_, data_.get()); 
} 
 
template<>
Vector_Base<std::complex<double>>::Vector_Base( const int& size ) : size_(2*size){} 
  
template<>
Vector_Base<std::complex<double>>::Vector_Base( const int& size, std::unique_ptr<double[]>& init_data ) :
   size_(size*2), data_( std::make_unique<double[]>(size_)) {
   std::copy_n( init_data.get(), size_, data_.get());   
}

template class Vector_Base<double>;
template class Vector_Base<std::complex<double>>;
