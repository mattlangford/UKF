#include <iostream>
#include <Eigen/Dense>
#include <atomic>
#include <memory>

#include "ukf.hh"
#include "sensors.hh"

using namespace ukf;

// void transform(SigmaPoints &points, const double dt)
// {
//     for (State &point : points)
//     {
//         point.position += dt * point.velocity;
//     }
// }

int main()
{


    // ukf_params_t params;
    // State initial_state;
    // Covariance initial_covariance = Covariance::Identity() * 1E-2;

    // initial_state.position << 1, 0, 0;
    // initial_state.velocity << 1, 0, 0;

    // double dt = 1;

    // UKF kf(initial_state, initial_covariance, params);
    // StateAndCovariance out = kf.unscented_transform(initial_state, initial_covariance, 
    //         std::bind(transform, std::placeholders::_1, dt));

    // std::cout << "State: " << std::endl;
    // out.state.print();
    // std::cout << "Covariance: " << std::endl;
    // std::cout << out.covariance << std::endl;
}
