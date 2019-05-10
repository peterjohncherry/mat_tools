#include "zmatrix.h"
#include "rvector.h"
#include "zvector.h"

using namespace std;

int main() {

  int nrows = 4;
  int ncols = 5;
  double value = 4.0;

  cout << "real_a " << endl;
  unique_ptr<RMatrix> real_a = make_unique<RMatrix>( nrows, ncols); 
  real_a->set_test_elems();
  real_a->print();

  cout << "real_b " << endl;
  unique_ptr<RMatrix> real_b = make_unique<RMatrix>( nrows, ncols); 
  real_b->set_test_elems();
  real_b->hconj();
  real_b->scale(2.0);
  real_b->print();

  cout << "real_c " << endl;
  unique_ptr<RMatrix> real_c = real_a->multiply(real_b);
  real_c->print();

  cout << "diagonalizing real_c" << endl;
  real_c->diagonalize(); 

  cout << "real_d" << endl;
  unique_ptr<RMatrix> real_d = *real_a + real_a;
  real_d->print();

  unique_ptr<RVector> real_v_a = make_unique<RVector>(nrows);

  {

  unique_ptr<ZMatrix> cplx_a = make_unique<ZMatrix>( nrows, ncols ); 
  cplx_a->set_test_elems();
  cplx_a->print();

  unique_ptr<ZMatrix> cplx_b = make_unique<ZMatrix>( nrows, ncols ); 
  cplx_b->set_test_elems();
  cplx_b->hconj();
  cplx_b->scale(2.0);
  cplx_b->print();

  cout << "zmatrix c = a.b" << endl;
  unique_ptr<ZMatrix> cplx_c = cplx_a->multiply(cplx_b);
  cplx_c->print();

  cout << "zmatrix d = b.a" << endl;
  unique_ptr<ZMatrix> cplx_d = cplx_b->multiply(cplx_a);
  cplx_d->print();

  cplx_d->generate_stdcomplex_data();
  cplx_d->generate_real_format_data();
  
  cplx_d->diagonalize_stdcomplex_routine();
  cout << "out " << endl;

  //const unique_ptr<RMatrix>& comb_d = cplx_d->combined_mat();
  cplx_d->combined_mat_->diagonalize();

  }
  
  {
  int length = 8;
  unique_ptr<double[]> init_data1 = make_unique<double[]>(length);
  unique_ptr<double[]> init_data2 = make_unique<double[]>(length);
  double* ptr1 = init_data1.get();
  double* ptr2 = init_data2.get();
  for ( int count = 0; count != length; ++ptr1, ++count, ++ptr2 ){
     *ptr1 = (double)(count);
     *ptr2 = (double)(count)+((double)(count)/2);
  }

  unique_ptr<RVector> testrvec1 = make_unique<RVector>(8, init_data1);
  unique_ptr<RVector> testrvec2 = make_unique<RVector>(8, init_data2);
  testrvec1->print();
  testrvec2->print();
  cout << endl;
  
  cout << "dot = " << testrvec1->dot_product(*testrvec2) << endl;
  cout << " norm v1 = " << testrvec1->norm() << endl;
  
  unique_ptr<ZVector> testzvec1 = make_unique<ZVector>(8, init_data1, init_data2);
  unique_ptr<ZVector> testzvec2 = make_unique<ZVector>(8, init_data2, init_data1);
  testzvec1->print();
  testzvec2->print();
  cout << " norm v1 = " << testzvec1->norm() << endl;
  cout << "dot = " << testzvec1->dot_product(*testzvec2) << endl;
  }

  return 0;
}
