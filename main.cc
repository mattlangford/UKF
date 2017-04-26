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


    ukf_params_t params;
    State initial_state;
    Covariance initial_covariance = Covariance::Identity() * 1E-2;

    initial_state.position << 1, 0, 0;
    initial_state.velocity << 1, 0, 0;

    StateAndCovariance initial;
    initial.state = initial_state;
    initial.covariance = initial_covariance;

    UKF kf({}, params);

    StateAndCovariance output = kf.update(initial, Clock::now());

    output.state.print();
}
