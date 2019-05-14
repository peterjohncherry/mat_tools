#ifndef __DAVIDSON_H
#define __DAVIDSON_H
#include "zmatrix.h"
#include "zvector.h"
#include "solver.h"

class Davidson : public Solver<double> {

  public :
 
    std::unique_ptr<RMatrix> mat_;

    void solve();
    
    Davidson(int max_it, std::string name): Solver(max_it, name) {
      std::cout << "Initialized solver " << name_ << std::endl;         
    }
    ~Davidson(){};
};

#endif
