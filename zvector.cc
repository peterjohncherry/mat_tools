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

unique_ptr<ZVector> ZVector::ax_plus_b( const ZVector& yy, std::complex<double> factor ) {

  unique_ptr<ZVector> vec_out = make_unique<ZVector>(yy.size());

  {
  unique_ptr<RVector> tmp2;
  {
  unique_ptr<RVector> tmp = real_vec_->ax_plus_b( *(yy.real_vec_), factor.real() ); 
  tmp2 = tmp->ax_plus_b( *(yy.imag_vec_), -factor.imag() ); 
  }
  vec_out->real_vec_ = real_vec_->ax_plus_b( *tmp2 , 1.0 ); 
  }
  
  {
  unique_ptr<RVector> tmp2;
  {
  unique_ptr<RVector> tmp = real_vec_->ax_plus_b( *(yy.real_vec_), factor.imag() ); 
  tmp2 = tmp->ax_plus_b( *(yy.imag_vec_), factor.real() ); 
  }
  vec_out->imag_vec_ = imag_vec_->ax_plus_b( *tmp2 , 1.0 ); 
  }  


  return vec_out;
}

void ZVector::scale( std::complex<double> factor ) {

  unique_ptr<ZVector> vec_out = make_unique<ZVector>(size_);

  {
    unique_ptr<RVector> tmp2 = make_unique<RVector>(*real_vec_);
    tmp2->scale( factor.real() );
    unique_ptr<RVector> tmp1 = make_unique<RVector>(*imag_vec_);
    real_vec_= tmp2->ax_plus_b( *tmp1, -factor.imag() );
  }

  {
    unique_ptr<RVector> tmp2 = make_unique<RVector>(*imag_vec_);
    tmp2->scale( factor.real() );
    unique_ptr<RVector> tmp1 = make_unique<RVector>(*real_vec_);
    imag_vec_= tmp2->ax_plus_b( *tmp1, factor.imag() );
  }
}
