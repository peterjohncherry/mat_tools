#include "zmatrix.h"

using namespace std;

int main() {

  int nrows = 3;
  int ncols = 4;
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
  cplx_d->diagonalize_stdcomplex_routine();
  cout << "out " << endl;

  }



  return 0;
}
