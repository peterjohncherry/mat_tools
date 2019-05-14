#include "davidson.h"
#include <vector>

void Davidson::solve(){

  int nvecs = 4; 

  std::vector<std::unique_ptr<RVector>> w_list;
  std::vector<std::unique_ptr<RVector>> u_list;

  std::unique_ptr<RMatrix> b_mat = std::make_unique<RMatrix>(1,1);
  for (int jj = 0; jj != nvecs; ++jj ) {
    
    std::unique_ptr<RVector> uj = mat_->extract_column(jj);
    uj->normalize();
    u_list.push_back( std::move(uj) );
    w_list.push_back(std::make_unique<RVector>());
    w_list[jj] = mat_->multiply( u_list[jj] );

    for ( int  kk = 0 ; kk != (jj+1); ++kk ) {
      std::cout << "kk = " << kk << std::endl; 
    }
    //std::unique_ptr<RMatrix> b_mat = std::make_unique<RMatrix>( std::move(b_mat), jj, jj );

  }

}
