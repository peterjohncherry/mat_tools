#include "zvector.h" 

#include <numeric>
using namespace std;

template<typename DataType>
void print_array(DataType* x, const int& count, string name = "") {

   cout << endl;
   if ( name != "")
     cout << name << endl;

   for (int ii = 0; ii != count; ++ii, ++x ) {
     cout << *x << " "; cout.flush();
   }
   cout << endl;
}

ZVector::ZVector( const int& size ) 
            : Vector_Base<std::complex<double>>( size ) { 
  data_ = std::make_unique<double[]>(size_*2);
  data_ptr_= data_.get();
}

ZVector::ZVector(const int& size, const std::complex<double>& init_val ) 
            : Vector_Base<std::complex<double>>( size ) { 
  data_ = std::make_unique<double[]>(size_*2);
  data_ptr_= data_.get();
  std::fill_n( std::fill_n( data_.get(), size_, init_val.real() ), size_, init_val.imag() ) ;
}

ZVector::ZVector( const int& size, const std::unique_ptr<double[]>& init_data ) 
            : Vector_Base<std::complex<double>>( size ) { 
  data_ = std::make_unique<double[]>(size_*2);
  data_ptr_= data_.get(); 
  std::copy_n( init_data.get(), size_*2, data_.get());
}

ZVector::ZVector( ZVector& vec) 
            : Vector_Base<std::complex<double>>( vec.size() ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
  std::copy_n(vec.data_ptr(), size_, data_ptr_);
}

std::complex<double> ZVector::dot_product( const ZVector& vec ) const {
   
   double real_part = std::inner_product(data_ptr(), data_ptr()+size_/2, vec.data_ptr(), double(0.0) );
   real_part = std::inner_product(data_ptr()+size_/2, data_ptr()+size_, vec.data_ptr()+size_/2, real_part );
   
   double imag_part = std::inner_product(data_ptr(), data_ptr()+size_/2, vec.data_ptr()+size_/2, double(0.0) );
   imag_part = std::inner_product(data_ptr()+size_/2, data_ptr()+size_, vec.data_ptr()+size_, imag_part );

   return std::complex<double>(real_part, imag_part);
}

void ZVector::print() {

   for ( auto ii = 0; ii != size_; ii++ ){
       cout << element(ii) << " "; cout.flush();
   }
   cout << endl;

}
