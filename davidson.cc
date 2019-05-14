#include "davidson.h"

void Davidson::solve(){

  std::cout << "hello" << std::endl; 

  std::unique_ptr<RVector> u1 = mat_->extract_column(1); 

  std::cout << "u1" << std::endl;
  u1->print();

  std::cout << "u1->norm() = " << u1->norm() << std::endl;

  std::cout << "normalizing" << std::endl;
  u1->normalize();

  std::cout << "after normalization" << std::endl;
  u1->print();
  std::cout << "norm = " <<  u1->norm() << std::endl;

}
