#include "rmatrix.h" 
#include "/home/peter/UTILS/LAPACK-3.8.0/LAPACKE/include/lapacke.h"

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
RMatrix::RMatrix(int nrows, int ncols) 
            : Matrix_Base<double>(nrows, ncols ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
}

RMatrix::RMatrix(int nrows, int ncols, const double& init_val ) 
            : Matrix_Base<double>(nrows, ncols ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
  std::fill_n( data_.get(), size_, init_val );
}

RMatrix::RMatrix(int nrows, int ncols, const std::unique_ptr<double[]>& init_data ) 
            : Matrix_Base<double>(nrows, ncols ) { 

  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get(); 
  std::copy_n( init_data.get(), size(), data_.get());
}

RMatrix::RMatrix( RMatrix& mat) 
            : Matrix_Base<double>(mat.nrows_, mat.ncols_ ) { 
  data_ = std::make_unique<double[]>(size_);
  data_ptr_= data_.get();
  std::copy_n(mat.data_ptr_, size_, data_ptr_);
}

void RMatrix::set_test_elems(){ 

  for ( int ii = 0; ii != nrows_; ++ii ) 
    for ( int jj = 0; jj != ncols_; ++jj ) {
      *(element_ptr(ii,jj)) = (double)(ii*10+jj);
    } 
}

void RMatrix::transpose(){ 

   unique_ptr<RMatrix> new_mat = make_unique<RMatrix>( ncols_, nrows_ );

   for ( int ii = 0; ii != nrows_; ++ii ) 
     for ( int jj = 0; jj != ncols_; ++jj )
       *(new_mat->element_ptr(jj,ii)) = element(ii,jj);

   data_ = std::move(new_mat->data_);
   data_ptr_ = data_.get();
   nrows_ = new_mat->nrows_;
   ncols_ = new_mat->ncols_;
}

unique_ptr<RMatrix>
RMatrix::ax_plus_b( const  unique_ptr<RMatrix>& matrix_b, double factor ) {
 
  assert( nrows_ == matrix_b->nrows() );
  assert( ncols_ == matrix_b->ncols() );
  
  unique_ptr<RMatrix> matrix_c = make_unique<RMatrix>(*this);

  int N = size_;
  double DA = factor;
  double* DX = matrix_b->data_ptr_;
  double* DY = matrix_c->data_ptr_;
  int INCX = 1;
  int INCY = 1;

  daxpy_( &N, &DA, DX, &INCX, DY, &INCY );

  return matrix_c;
}

void RMatrix::scale( double factor ){

   double* tmp = data_ptr_;
   for ( int  ii = 0 ; ii != size_ ; ++ii, ++tmp)
     *tmp = (*tmp)*factor;

}

std::unique_ptr<RMatrix> RMatrix::multiply( const unique_ptr<RMatrix>& matrix_b ) {
  
  char TRANSA = 'N';
  char TRANSB = 'N';
  double ALPHA = 1.0;
  double BETA  = 0.0;
  double* A = data_ptr_;
  double* B = matrix_b->data_ptr_;

  int M = nrows_;
  int N = matrix_b->ncols_;
  int K = ncols_;
  int LDA = M;
  int LDB = K;
  int LDC = N;
  
  unique_ptr<RMatrix> matrix_c = make_unique<RMatrix>( M, N, 0.0 );
  double* C = matrix_c->data_ptr_;
 
  dgemm_( &TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC );

  return matrix_c;
}

unique_ptr<RVector> 
RMatrix::matvec_mult_lapack( unique_ptr<RVector>& vec ){

  unique_ptr<RVector> vec_out = make_unique<RVector>(nrows_, 0.0);
  char TRANS = 'N';
  double ALPHA = 1.0;
  double BETA  = 0.0;

  double* A = data_ptr_;
  double* X = vec->data_ptr();
  double* Y = vec_out->data_ptr();
 
  int M = nrows_;
  int N = ncols_;
  int LDA = nrows_;
  int INCX = 1;
  int INCY = 1;

  dgemv_( &TRANS, &M, &N, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY );
  return vec_out; 
}

void RMatrix::diagonalize(){

  char BALANC = 'N';
  char JOBVL = 'V'; // Set V to computes right eigenvalues, N does not.
  char JOBVR = 'V'; // Set V computes left eigenvalues, N does not.
  char SENSE = 'N'; // do not compute conditioning numbers

  if ( SENSE == 'E' || SENSE == 'B' ) 
    if ( JOBVL != 'V' || JOBVR != 'V' ) 
       throw std::logic_error( " incosistent definition of JOBVL/JOBVR/SENSE in eigval routine ");

  int N = ncols_;
  double* A = data_ptr_;
  int LDA = nrows_;
  eigenvalues_real_ = make_unique<double[]> (N);
  eigenvalues_imag_ = make_unique<double[]> (N);
  double* WR = eigenvalues_real_.get();
  double* WI = eigenvalues_imag_.get();

  int LDVL = (JOBVL == 'V' ? N : 0 ); 
  double* VL;
  if ( JOBVL == 'V' ){ 
    l_eigenvectors_ = make_unique<RMatrix>( N, LDVL, 0.0);
    VL = l_eigenvectors_->data_ptr();
  }

  int LDVR = (JOBVR == 'V' ? N : 0 ); 
  double* VR;
  unique_ptr<double[]> r_eigenvectors_data;
  if ( JOBVR == 'V' ) {
    r_eigenvectors_ =  make_unique<RMatrix>( LDVR, N, 0.0 );
    VR = r_eigenvectors_->data_ptr();
  }
  int ILO = 0;
  int IHI = 0;

  unique_ptr<double[]> SCALE_array = make_unique<double[]>(N);
  double* SCALE = SCALE_array.get();

  int INFO = 0;
  double ABNORM;

  unique_ptr<double[]> RCONDE_array = make_unique<double[]>(N);
  std::fill_n(RCONDE_array.get(), N, 0.0);
  double* RCONDE = RCONDE_array.get();

  unique_ptr<double[]> RCONDV_array = make_unique<double[]>(N);
  std::fill_n(RCONDV_array.get(), N, 0.0);
  double* RCONDV = RCONDV_array.get();

  int LWORK = 3*N;
  unique_ptr<double[]> WORK_array = make_unique<double[]>(LWORK);
  std::fill_n(WORK_array.get(), LWORK, 0.0);
  double* WORK = WORK_array.get();

  double ABNRM = 0.0;
  int* IWORK;
  unique_ptr<int[]> IWORK_array;

  if ( SENSE != 'N' && SENSE != 'E' ) {
    IWORK_array =  make_unique<int[]>(2*(N -1));
    std::fill_n(IWORK_array.get(), N, 0);
    IWORK = IWORK_array.get();
  }
 
  dgeevx_( &BALANC, &JOBVL, &JOBVR, &SENSE, &N,   // args 1-5
           A, &LDA, WR, WI, VL,                   // args 6-10
           &LDVL, VR, &LDVR, &ILO, &IHI,          // args 11-15
           SCALE, &ABNRM, RCONDE, RCONDV, WORK,   // args 16-20
           &LWORK, IWORK, &INFO );                // args 21-23

  print_array(WR, N, "real_part_of_eigenvalues"); cout << endl;  
  print_array(WI, N, "imag_part_of_eigenvalues"); cout << endl;  

  cout << "right eigenvectors " << endl;
  r_eigenvectors_->print();

  cout << "left eigenvectors " << endl;
  l_eigenvectors_->print();

}

void RMatrix::symmetrize(){
  cout << " void RMatrix::symmetrize() " << endl;
  unique_ptr<RMatrix> trans_mat = make_unique<RMatrix>(*this);
  scale(0.5);
  ax_plus_b( trans_mat, 0.5 );
  this->print();

}

void RMatrix::print() {

   for ( auto ii = 0; ii != nrows_; ii++ ){
     for ( auto jj = 0; jj != ncols_; jj++ )
       cout << element(ii,jj) << " "; cout.flush();
     cout << endl;
   }
   cout << endl;

}
