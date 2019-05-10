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
//   real_vec_ = make_unique<RVector>(size_);
//   imag_vec_ = make_unique<RVector>(size_);
}

ZVector::ZVector(const int& size, const std::complex<double>& init_val ) 
            : Vector_Base<std::complex<double>>( size ) { 
  real_vec_ = make_unique<RVector>(size, init_val.real() );
  imag_vec_ = make_unique<RVector>(size, init_val.imag() );   
}

ZVector::ZVector( const int& size, const std::unique_ptr<double[]>& init_real_data, 
                                   const std::unique_ptr<double[]>& init_imag_data ) 
            : Vector_Base<std::complex<double>>( size ) { 
   real_vec_ = make_unique<RVector>(size, init_real_data );
   imag_vec_ = make_unique<RVector>(size, init_imag_data );
}

ZVector::ZVector( const int& size, const double* init_real_data, const double* init_imag_data ) 
            : Vector_Base<std::complex<double>>( size ) { 
//   real_vec_ = make_unique<RVector>(size, init_real_data );
//   imag_vec_ = make_unique<RVector>(size, init_imag_data );
}

ZVector::ZVector(const  ZVector& vec) 
            : Vector_Base<std::complex<double>>( vec.size() ) { 

   real_vec_ = make_unique<RVector>(*(vec.real_vec_));
   imag_vec_ = make_unique<RVector>(*(vec.imag_vec_));
}

std::complex<double> ZVector::dot_product( const ZVector& vec ) const {
   
   double real_part = real_vec_->dot_product( *(vec.real_vec_));
   real_part -= imag_vec_->dot_product( *(vec.imag_vec_));

   double imag_part = real_vec_->dot_product( *(vec.imag_vec_));
   imag_part += imag_vec_->dot_product( *(vec.real_vec_));

   return std::complex<double>(real_part, imag_part);
}

double ZVector::norm() const {
   return ( real_vec_->dot_product( *real_vec_) + imag_vec_->dot_product( *imag_vec_) );
}
void ZVector::print() const {

   for ( auto ii = 0; ii != size_; ii++ ){
       cout << element(ii) << " "; cout.flush();
   }
   cout << endl;

}
