#define BOOST_TEST_MODULE Sensor Tests
#include <boost/test/included/unit_test.hpp>

#include "../sensors.hh"
#include "../utils.hh"

using namespace ukf;

#define A_TOL 1E-5
#define EIGEN_CLOSE(lhs, rhs) ((lhs - rhs).norm() < A_TOL)

BOOST_AUTO_TEST_CASE(No_Noise_Accelerometer_Test)
{
    for (size_t i = 0; i < 5; ++i)
    {
        Eigen::Vector3d acceleration = Eigen::Vector3d::Random();
        Covariance covariance = Covariance::Identity() * Covariance::Random().cwiseAbs();

        State true_state;
        true_state.acceleration = acceleration;

        SigmaPoints sigmas = ukf_utils::compute_sigma_points(true_state, covariance);

        Accelerometer a;
        ObsCovCrossCov output = a.compute_observation(sigmas, true_state);

        BOOST_REQUIRE(EIGEN_CLOSE(true_state.acceleration, acceleration));
        // TODO: Check covariances
    }
}

BOOST_AUTO_TEST_CASE(No_Noise_Gyroscope_Test)
{
    for (size_t i = 0; i < 5; ++i)
    {
        Eigen::Vector3d angular_vel = Eigen::Vector3d::Random();
        Covariance covariance = Covariance::Identity() * Covariance::Random().cwiseAbs();

        State true_state;
        true_state.angular_vel = angular_vel;

        SigmaPoints sigmas = ukf_utils::compute_sigma_points(true_state, covariance);

        Gyroscope g;
        ObsCovCrossCov output = g.compute_observation(sigmas, true_state);

        BOOST_REQUIRE(EIGEN_CLOSE(true_state.angular_vel, angular_vel));
        // TODO: Check covariances
    }
}
