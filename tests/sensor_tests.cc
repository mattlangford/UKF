#define BOOST_TEST_MODULE Sensor Tests
#include <boost/test/included/unit_test.hpp>

#include "../types.hh"
#include "../sensors.hh"

using namespace ukf;

#define A_TOL 1E-5
#define EIGEN_CLOSE(lhs, rhs) ((lhs - rhs).norm() < A_TOL)

//
// TODO: Move this to utils and use it there instead of in the UKF class
//
SigmaPoints compute_sigma_points(const State& state,
                                 const Covariance& covariance,
                                 const ukf_params_t& params)

{
    //
    // Let's do the transform now (not really covariance type, but close enough)
    //
    Covariance cov_sqrt = covariance.llt().matrixL();
    cov_sqrt *= params.sqrt_cov_factor;

    //
    // Populate sigma vector
    //
    SigmaPoints sigmas(NUM_SIGMA_POINTS);
    sigmas[0] = state;
    for (size_t i = 0; i < NUM_STATES; ++i)
    {
        //
        // Since sigma[0] is taken, add one to the index
        //
        sigmas[2 * i + 1] = state + cov_sqrt.row(i);
        sigmas[2 * i + 2] = state + -cov_sqrt.row(i);
    }

    return sigmas;
}

bool states_are_close(const State& lhs, const State& rhs)
{
    return EIGEN_CLOSE(lhs.position, rhs.position) &&
           EIGEN_CLOSE(lhs.velocity, rhs.velocity) &&
           EIGEN_CLOSE(lhs.acceleration, rhs.acceleration) &&
           EIGEN_CLOSE(lhs.orientation, rhs.orientation) &&
           EIGEN_CLOSE(lhs.angular_vel, rhs.angular_vel) &&
           EIGEN_CLOSE(lhs.angular_acc, rhs.angular_acc) &&
           EIGEN_CLOSE(lhs.acc_bias, rhs.acc_bias) &&
           EIGEN_CLOSE(lhs.gyro_bias, rhs.gyro_bias);
}

BOOST_AUTO_TEST_CASE(No_Noise_Accelerometer_Test)
{
    ukf_params_t params;

    for (size_t i = 0; i < 5; ++i)
    {
        Eigen::Vector3d acceleration = Eigen::Vector3d::Random();
        Covariance covariance = Covariance::Identity() * Covariance::Random().cwiseAbs();

        State true_state;
        true_state.acceleration = acceleration;

        SigmaPoints sigmas = compute_sigma_points(true_state, covariance, params);

        Accelerometer a;
        ObsCovCrossCov output = a.compute_observation(sigmas, true_state, params);

        BOOST_REQUIRE(EIGEN_CLOSE(true_state.acceleration, acceleration));
        // TODO: Check covariances
    }
}

BOOST_AUTO_TEST_CASE(No_Noise_Gyroscope_Test)
{
    ukf_params_t params;

    for (size_t i = 0; i < 5; ++i)
    {
        Eigen::Vector3d angular_vel = Eigen::Vector3d::Random();
        Covariance covariance = Covariance::Identity() * Covariance::Random().cwiseAbs();

        State true_state;
        true_state.angular_vel = angular_vel;

        SigmaPoints sigmas = compute_sigma_points(true_state, covariance, params);

        Gyroscope g;
        ObsCovCrossCov output = g.compute_observation(sigmas, true_state, params);

        BOOST_REQUIRE(EIGEN_CLOSE(true_state.angular_vel, angular_vel));
        // TODO: Check covariances
    }
}
