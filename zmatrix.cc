#include "zmatrix.h"
#include "/home/peter/UTILS/LAPACK-3.8.0/LAPACKE/include/lapacke.h"

using namespace std;

#ifndef LAPACK_CCGEVZZZZ
#define LAPACK_CCGEVZZZZ

template<typename DataType>
void print_array(DataType* x, int count, string& name = "") {

   cout << endl;
   if ( name != "")
     cout << name << endl;

   for (int ii = 0; ii != count; ++ii, ++x ) {
     cout << *x << " "; cout.flush();
   }
   cout << endl;
}

template<typename DataType>
void print_array ( DataType* dd, int size, string name = "" ) { 


   cout << name << endl;
   int ii = 0; 
   while ( ii != size ) {
     cout << *dd << " " ; cout.flush();
     ++ii;
   }

}

extern "C" {
  

  extern void zggev_(  char* JOBVL /*1*/, char* JOBVR/*2*/, lapack_int* N/*3*/, lapack_complex_double* A/*4*/, lapack_int* LDA/*5*/,
                      lapack_complex_double* B/*6*/, lapack_int* LDB/*7*/, lapack_complex_double* ALPHA/*8*/, lapack_complex_double* BETA/*9*/,
                      lapack_complex_double* VL/*10*/, lapack_int* LDVL/*11*/, lapack_complex_double* VR/*12*/, lapack_int* LDVR/*13*/,
                      lapack_complex_double* WORK/*14*/, lapack_int* LWORK/*15*/, double* RWORK/*16*/, lapack_int* INFO/*17*/ );
}
#endif

ZMatrix::ZMatrix( int ncols, int nrows ): Matrix_Base<complex<double>>( nrows, ncols ) {
  real_mat_ = std::make_unique<RMatrix>(nrows, ncols);
  imag_mat_ = std::make_unique<RMatrix>(nrows, ncols);

};

ZMatrix::ZMatrix( int ncols, int nrows, const complex<double>& init_val ) 
              : Matrix_Base<complex<double>>( nrows, ncols ) {
  real_mat_ = std::make_unique<RMatrix>(nrows, ncols, init_val.real() );
  imag_mat_ = std::make_unique<RMatrix>(nrows, ncols, init_val.imag() );

};

ZMatrix::ZMatrix( int ncols, int nrows, const std::unique_ptr<complex<double>[]>& init_data ) 
              : Matrix_Base<complex<double>>( nrows, ncols ) {

  std::unique_ptr<double[]> real_data = make_unique<double[]>(nrows*ncols);
  std::unique_ptr<double[]> imag_data = make_unique<double[]>(nrows*ncols);

  double* r_ptr = real_data.get();
  double* i_ptr = imag_data.get();
  std::complex<double>* c_ptr = init_data.get();

  for ( int ii = 0; ii != nrows*ncols; ++ii, ++r_ptr, ++i_ptr, ++c_ptr ){
     *r_ptr = (*c_ptr).real();
     *i_ptr = (*c_ptr).imag();
  } 

  real_mat_ = std::make_unique<RMatrix>(nrows, ncols, real_data);
  imag_mat_ = std::make_unique<RMatrix>(nrows, ncols, imag_data);
  

};

ZMatrix::ZMatrix( int ncols, int nrows, const std::unique_ptr<double[]>& init_data ) 
              : Matrix_Base<complex<double>>( nrows, ncols ) {

  std::unique_ptr<double[]> real_data = make_unique<double[]>(nrows*ncols);
  std::unique_ptr<double[]> imag_data = make_unique<double[]>(nrows*ncols);

  double* r_ptr = real_data.get();
  double* i_ptr = imag_data.get();
  double* c_ptr = init_data.get();

  for ( int ii = 0; ii != nrows*ncols; ++ii, ++r_ptr, ++i_ptr, ++c_ptr ){
     *r_ptr = *c_ptr;
      ++c_ptr;
     *i_ptr = *c_ptr;
  } 

  real_mat_ = std::make_unique<RMatrix>(nrows, ncols, real_data );
  imag_mat_ = std::make_unique<RMatrix>(nrows, ncols, imag_data );

};

void ZMatrix::set_test_elems(){ 

  for ( int ii = 0; ii != nrows_; ++ii ){ 
    for ( int jj = 0; jj != ncols_; ++jj ){
      *(real_mat_->element_ptr(ii,jj)) = (double)(ii);
      *(imag_mat_->element_ptr(ii,jj)) = (double)(jj);
    }
  }
}

void ZMatrix::transpose() {

   real_mat_->transpose();
   imag_mat_->transpose();
   int buff = ncols_;
   ncols_ = nrows_;
   nrows_ = buff; 

}

void ZMatrix::hconj() {
   int buff = ncols_;
   ncols_ = nrows_;
   nrows_ = buff; 

   real_mat_->transpose();
   imag_mat_->transpose();
   imag_mat_->scale(-1.0);

}

std::unique_ptr<ZMatrix> ZMatrix::ax_plus_b( const std::unique_ptr<ZMatrix>& matrix_b, double factor ) {

   unique_ptr<ZMatrix> new_mat = make_unique<ZMatrix>(nrows_, ncols_);

   new_mat->real_mat_ = std::move(real_mat_->ax_plus_b(matrix_b->real_mat_ , factor ));
   new_mat->imag_mat_ = std::move(imag_mat_->ax_plus_b(matrix_b->imag_mat_ , factor ));

   return std::move(new_mat);
}

std::unique_ptr<ZMatrix> ZMatrix::ax_plus_b( const std::unique_ptr<ZMatrix>& matrix_b, complex<double> factor ) {

   unique_ptr<ZMatrix> new_mat = make_unique<ZMatrix>(nrows_, ncols_);

   new_mat->real_mat_ = std::move(real_mat_->ax_plus_b(matrix_b->real_mat_ , factor.real() ));
   new_mat->real_mat_ = std::move(new_mat->real_mat_->ax_plus_b(matrix_b->imag_mat_ ,-(factor.imag()) ));

   new_mat->imag_mat_ = std::move(imag_mat_->ax_plus_b(matrix_b->real_mat_ , factor.imag() ));
   new_mat->imag_mat_ = std::move(new_mat->imag_mat_->ax_plus_b(matrix_b->imag_mat_ , factor.real() ));

   return std::move(new_mat);
}

void ZMatrix::scale( double factor ){
 
    real_mat_->scale(factor);
    imag_mat_->scale(factor);

};

void ZMatrix::scale( std::complex<double> factor ){

    real_mat_->scale(factor.real());
    unique_ptr<RMatrix> new_real_mat = real_mat_->ax_plus_b(imag_mat_, -factor.imag());

    imag_mat_->scale(factor.real());
    unique_ptr<RMatrix> new_imag_mat = imag_mat_->ax_plus_b(real_mat_, factor.imag()/factor.real());

    real_mat_ = std::move(new_real_mat);
    imag_mat_ = std::move(new_imag_mat);
};

void ZMatrix::print() {
cout << "ZMatrix::print() " << endl;

   for ( auto ii = 0; ii != nrows_; ii++ ){
     for ( auto jj = 0; jj != ncols_; jj++ ){
       cout << element(ii,jj) << " "; cout.flush();
     }
     cout << endl;
   }
   cout << endl;

}

std::unique_ptr<ZMatrix> ZMatrix::multiply( std::unique_ptr<ZMatrix>& matb ) {
   cout << "orig mat in ZMatrix::multiply " << endl;
   unique_ptr<ZMatrix> matc = make_unique<ZMatrix>(nrows_, matb->ncols_);
   
   {
     unique_ptr<RMatrix> tmp =  real_mat_->multiply( matb->real_mat_ );
     matc->real_mat_ = tmp->ax_plus_b(imag_mat_->multiply( matb->imag_mat_ ), -1.0 );
   } 
   {
     unique_ptr<RMatrix> tmp =  real_mat_->multiply( matb->imag_mat_ );
     matc->imag_mat_ = tmp->ax_plus_b(imag_mat_->multiply( matb->real_mat_ ), 1.0 );
   }
   return matc;

}

void ZMatrix::generate_complex_data(){

   complex_data_ = make_unique<double[]>( size_*2 );
   double* c_ptr = complex_data_.get();
   double* r_ptr = real_mat_->data_ptr();
   double* i_ptr = imag_mat_->data_ptr();

   for ( int ii = 0 ; ii != size_; ++ii, ++c_ptr, ++r_ptr, ++i_ptr ) {
     *c_ptr = *r_ptr;
     ++c_ptr; 
     *c_ptr = *i_ptr;
   }

}

void ZMatrix::generate_stdcomplex_data(){

   stdcomplex_data_ = make_unique<std::complex<double>[]>( size_);
   std::complex<double>* c_ptr = stdcomplex_data_.get();
   double* r_ptr = real_mat_->data_ptr();
   double* i_ptr = imag_mat_->data_ptr();

   for ( int ii = 0 ; ii != size_; ++ii, ++c_ptr, ++r_ptr, ++i_ptr ) 
     *c_ptr = std::complex<double> (*r_ptr, *i_ptr);

}

void diagonalize_complex_routine(std::unique_ptr<ZMatrix>& mat ) {
  
   char JOBVL = 'N';
   char JOBVR = 'N';
   int N = mat->ncols();
   double* A = mat->complex_data_ptr();
   int LDA = mat->nrows();
   int LDB = mat->nrows();

   unique_ptr<double[]> B_array = make_unique<double[]>(LDB*N*2);
   double* B = B_array.get();

   unique_ptr<double[]> ALPHA_array = make_unique<double[]>(N*2);
   double* ALPHA = ALPHA_array.get();

   unique_ptr<double[]> BETA_array = make_unique<double[]>(N*2);
   double* BETA = BETA_array.get();

   int LDVL = N;
   int LDVR = N;
   unique_ptr<double[]> VL_array = make_unique<double[]>(N*2);
   double* VL = VL_array.get();

   unique_ptr<double[]> VR_array = make_unique<double[]>(N*2);
   double* VR = VR_array.get();

   int LWORK = 3*N; // GET THIS PROPERLY

   unique_ptr<double[]> WORK_array = make_unique<double[]>(LWORK*2);
   double* WORK = WORK_array.get();

   unique_ptr<double[]> RWORK_array = make_unique<double[]>(8*N);
   double* RWORK = RWORK_array.get();
   int INFO = 0;

}

void ZMatrix::diagonalize_stdcomplex_routine() {
   
   char JOBVL = 'V';
   char JOBVR = 'V';
   int N = ncols_;
   lapack_complex_double* A = (lapack_complex_double*)(stdcomplex_data_.get());
   int LDA = nrows_;
   int LDB = nrows_;
   int LDVL = N;
   int LDVR = N;
   int LWORK = 3*N; // GET THIS PROPERLY
   
   unique_ptr<double[]> B_array = make_unique<double[]>(LDB*N*4);
   lapack_complex_double* B = (lapack_complex_double*) B_array.get();
   
   unique_ptr<double[]> ALPHA_array = make_unique<double[]>(N*2);
   lapack_complex_double* ALPHA = (lapack_complex_double*)ALPHA_array.get();
   
   unique_ptr<double[]> BETA_array = make_unique<double[]>(N*2);
   lapack_complex_double* BETA = (lapack_complex_double*)BETA_array.get();
  
   lapack_complex_double* VL;
   unique_ptr<double[]> VL_array;
   switch (JOBVL){ 
     case (int)'V' : 
       VL_array = make_unique<double[]>(N*2*LDVL);
       VL = (lapack_complex_double*)VL_array.get();
       break;
     case (int)'N' :
       break;
     default :
       throw std::logic_error("ZMatrix::diagonalize JOBVL must be either 'V' or 'N' !!");
   } 
 
   lapack_complex_double* VR;
   unique_ptr<double[]> VR_array;
   switch (JOBVR){ 
     case (int)'V' : 
       VR_array = make_unique<double[]>(N*2*LDVR);
       VR = (lapack_complex_double*)VR_array.get();
       break;
     case (int)'N' :
       break;
     default :
       throw std::logic_error("ZMatrix::diagonalize JOBVR must be either 'V' or 'N' !!");
   } 
   
   unique_ptr<double[]> WORK_array = make_unique<double[]>(LWORK*2);
   lapack_complex_double* WORK = (lapack_complex_double*)WORK_array.get();
   
   unique_ptr<double[]> RWORK_array = make_unique<double[]>(8*N*2);
   double* RWORK = RWORK_array.get();
   int INFO = 0;

   zggev_( &JOBVL, &JOBVR, &N, A, &LDA, B, &LDB, ALPHA, BETA, VL,
           &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO );
 
   switch (JOBVR){ 
     case (int)'V' : 
       r_eigenvectors_ = make_unique<ZMatrix>( nrows_, ncols_, VR_array );
       break;
     case (int)'N' :
       break;
     default :
       throw std::logic_error("ZMatrix::diagonalize JOBVR must be either 'V' or 'N' !!");
   } 

   switch (JOBVL){ 
     case (int)'V' : 
       l_eigenvectors_ = make_unique<ZMatrix>( nrows_, ncols_, VL_array );
       break;
     case (int)'N' :
       break;
     default :
       throw std::logic_error("ZMatrix::diagonalize JOBVR must be either 'V' or 'N' !!");
   } 
}

// Combines real and imag parts into single real matrix for use with eigensolvers, 
// [A+iB] -> [[ A  -B ]
//           [[ B   A ] 
void ZMatrix::generate_real_format_data(){

   unique_ptr<double[]> real_format_data = make_unique<double[]>(size_*4);
   double* r_ptr = real_mat_->data_ptr_;
   double* i_ptr = imag_mat_->data_ptr_;
   double* rf_ptr = real_format_data.get();

   //imag_mat_->scale(-1.0);
   for ( int ii = 0 ; ii != nrows_; ++ii, r_ptr+=ncols_, i_ptr += ncols_ ) {
      rf_ptr = std::copy_n(r_ptr, ncols_, rf_ptr);
      rf_ptr = std::copy_n(i_ptr, ncols_, rf_ptr);
   }

   imag_mat_->scale(-1.0);
   r_ptr = real_mat_->data_ptr();
   i_ptr = imag_mat_->data_ptr();

   for ( int ii = 0 ; ii != nrows_; ++ii, r_ptr+=ncols_, i_ptr += ncols_ ) {
      rf_ptr = std::copy_n(i_ptr, ncols_, rf_ptr);
      rf_ptr = std::copy_n(r_ptr, ncols_, rf_ptr);
   }

   combined_mat_ = make_unique<RMatrix>(nrows_*2, ncols_*2, std::move(real_format_data));
   cout << "combined mat" << endl;
   combined_mat_->print(); cout << endl;
}

std::unique_ptr<ZMatrix> ax_plus_b( const std::unique_ptr<ZMatrix>& matrix_b, std::complex<double> factor ) {

  

};
