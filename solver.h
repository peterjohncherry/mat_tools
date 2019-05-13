#ifndef __SOLVER_H
#define __SOLVER_H
#include <string>

class Solver {

  public : 

      int max_it_;
      std::string name_;

      virtual void solve(int max_it, std::string name ) {} 

      Solver(int max_it, std::string name ): max_it_(max_it), name_(name){}
      ~Solver(){};
 
};
#endif


