#include "rvector.h" 
#include "/home/peter/UTILS/LAPACK-3.8.0/LAPACKE/include/lapacke.h"
#include <numeric>

extern "C" {
extern int dgemm_(char * transa, char * transb, int * m, int * n, int * k,
                  double * alpha, double * A, int * lda,
                  double * B, int * ldb, double * beta,
                  double *, int * ldc); 

extern int dgemv_( char* transa, int* M, int* N, double* ALPHA,	double*	A, int* LDA, double* X, int* INCX,
		   double* BETA, double* Y, int* INCY );

extern int daxpy_( int* N, double* A, double* B, int* INCX, double* Y, int* INCY );
}

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

RVector::RVector( const int& size ) 
            : Vector_Base<double>( size ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
}

RVector::RVector(const int& size, const double& init_val ) 
            : Vector_Base<double>( size ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
  std::fill_n( data_.get(), size_, init_val );
}

RVector::RVector( const int& size, const std::unique_ptr<double[]>& init_data ) 
            : Vector_Base<double>( size ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get(); 
  std::copy_n( init_data.get(), size_, data_.get());
}

RVector::RVector( RVector& vec) 
            : Vector_Base<double>( vec.size() ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
  std::copy_n(vec.data_ptr(), size_, data_ptr_);
}

double RVector::dot_product( const RVector& vec ) const {

   return std::inner_product(data_ptr(), data_ptr()+size_, vec.data_ptr(), double(0.0) );

}

void RVector::print() {

   for ( auto ii = 0; ii != size_; ii++ ){
       cout << element(ii) << " "; cout.flush();
   }
   cout << endl;

}
