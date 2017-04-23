#include <iostream>
#include <Eigen/Dense>

#include "ukf.hh"


using namespace ukf;
int main()
{
    ukf_params_t params;    
    State initial_state;     
    Covariance initial_covariance = Covariance::Identity() * 1E-2;

    initial_state.position << 1, 0, 0;

    UKF kf(initial_state, initial_covariance, params);
    kf.test_UT(initial_state, initial_covariance);
}
