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

Eigen::Quaterniond exp_q(const Eigen::Vector3d &w)
{
    double theta = w.norm();
    if (theta < 1E-6)
    {
        return {1.0, 0.0, 0.0, 0.0};
    }

    Eigen::Quaterniond result;
    result.w() = cos(theta / 2.0);
    result.vec() = w * sin(theta / 2.0) / theta;
    return result;
}

Eigen::Vector3d ln_q(const Eigen::Quaterniond &q)
{
    Eigen::Vector3d result;
    double theta = acos(q.w());

    if (theta < 1E-6)
    {
        return {0.0, 0.0, 0.0};
    }

    return 2 * q.vec() * theta / sin(theta);
}

Eigen::Quaterniond average(const Eigen::Quaterniond &lhs, const Eigen::Quaterniond &rhs)
{
    //
    // Map each quaternion to it's Lie Algebra, do math there, then convert back
    //
    return exp_q((ln_q(lhs) + ln_q(rhs)) / 2.0);
}

int main()
{
    Eigen::Quaterniond one = {0.91071847, -0.13764163, -0.05819395,  0.3850455};
    Eigen::Quaterniond two = {0.89770629, -0.20183273, -0.03215254,  0.39032445};

    Eigen::Quaterniond q = average(one, two);

    std::cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;

    // ukf_params_t params;
    // State initial_state;
    // Covariance initial_covariance = Covariance::Identity() * 1E-2;

    // initial_state.position << 1, 0, 0;
    // initial_state.velocity << 1, 0, 0;

    // StateAndCovariance initial;
    // initial.state = initial_state;
    // initial.covariance = initial_covariance;

    // UKF kf({}, params);

    // StateAndCovariance output = kf.update(initial, Clock::now());

    // output.state.print();
}
