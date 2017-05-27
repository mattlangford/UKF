#define BOOST_TEST_MODULE Utils Tests
#include <boost/test/included/unit_test.hpp>
#include <Eigen/Dense>

#include "../utils.hh"
#include "../types.hh"

#define A_TOL 2E-4
#define EIGEN_CLOSE(lhs, rhs) ((lhs - rhs).norm() < A_TOL)

bool quat_are_close(const Eigen::Quaterniond &lhs, const Eigen::Quaterniond &rhs)
{
    return 2 * acos(clamp((rhs * lhs.inverse()).w(), -1.0, 1.0)) < A_TOL;
}

bool states_are_close(const ukf::State& lhs, const ukf::State& rhs)
{
    return EIGEN_CLOSE(lhs.position, rhs.position) &&
           EIGEN_CLOSE(lhs.velocity, rhs.velocity) &&
           EIGEN_CLOSE(lhs.acceleration, rhs.acceleration) &&
           quat_are_close(lhs.orientation, rhs.orientation) &&
           EIGEN_CLOSE(lhs.angular_vel, rhs.angular_vel) &&
           EIGEN_CLOSE(lhs.angular_acc, rhs.angular_acc) &&
           EIGEN_CLOSE(lhs.acc_bias, rhs.acc_bias) &&
           EIGEN_CLOSE(lhs.gyro_bias, rhs.gyro_bias);
}

BOOST_AUTO_TEST_CASE(Rotate_Vector_By_Quaternion)
{
    // I used numpy and tf.transformations to confirm these
    Eigen::Vector3d forward(1, 0, 0);
    Eigen::Quaterniond no_rotation(1, 0, 0, 0);

    BOOST_REQUIRE(EIGEN_CLOSE(forward, rotation_helpers::rotate_vec_by_quat(forward, no_rotation)));

    Eigen::Quaterniond x_90_z_90(0.5, 0.5, -0.5, 0.5);
    Eigen::Vector3d up = {0, 0, 1};
    BOOST_REQUIRE(EIGEN_CLOSE(up, rotation_helpers::rotate_vec_by_quat(forward, x_90_z_90)));

    Eigen::Quaterniond random_1(-0.431410340004, -0.227475712562, -0.620743435549, 0.613854629334); 
    Eigen::Vector3d expected_1(-0.52427984, -0.24723836, -0.81486431);
    BOOST_REQUIRE(EIGEN_CLOSE(expected_1, rotation_helpers::rotate_vec_by_quat(forward, random_1)));

    // Testing non normalized vectors
    Eigen::Vector3d random_vector(-0.60508391,  0.69961366,  1.22848347);
    Eigen::Quaterniond random_2(-0.325688076033, -0.783530090301, 0.201620620369, 0.489241249452);
    Eigen::Vector3d expected_2(-1.36749835, -0.49493513,  0.49974487);
    BOOST_REQUIRE(EIGEN_CLOSE(expected_2, rotation_helpers::rotate_vec_by_quat(random_vector, random_2)));
}

BOOST_AUTO_TEST_CASE(Quaterion_Inverse)
{
    Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
    BOOST_REQUIRE(quat_are_close(identity * identity.inverse(), identity));

    Eigen::Quaterniond random(-0.325688076033, -0.783530090301, 0.201620620369, 0.489241249452);
    BOOST_REQUIRE(quat_are_close(random * random.inverse(), identity));

    Eigen::Quaterniond rot(0.988771, 0, 0, 0.149438);
    Eigen::Quaterniond rot_inv(0.988771, 0, 0, -0.149438);
    BOOST_REQUIRE(quat_are_close(rot * rot_inv, identity));
}

BOOST_AUTO_TEST_CASE(Quaternion_Mappings)
{
    // Check that a tangent space vector can be mapped to a rotation and back
    Eigen::Vector3d random_tangent_vect = Eigen::Vector3d::Random();
    BOOST_REQUIRE(EIGEN_CLOSE(random_tangent_vect,
                rotation_helpers::ln(rotation_helpers::exp_q(random_tangent_vect))));

    // Check rotation matrix and quaternion map to the same vector, checked in python
    Eigen::Matrix3d random_rot_r;
    random_rot_r <<  0.73383118,  0.45137516,  0.50769308,
                     0.505302  , -0.8621843 ,  0.03616794,
                     0.45405031,  0.22999717, -0.8607785;
    Eigen::Quaterniond random_rot_q = {0.05212577, 0.92962277, 0.25727564,  0.25863808};

    Eigen::Vector3d map_to = {2.82738126,  0.78248547,  0.78662924};

    BOOST_REQUIRE(EIGEN_CLOSE(rotation_helpers::ln(random_rot_r), map_to));
    BOOST_REQUIRE(EIGEN_CLOSE(rotation_helpers::ln(random_rot_q), map_to));
}

BOOST_AUTO_TEST_CASE(Average_Rotations)
{
    // Trivial case, average should be the identity. Everything checked in python
    Eigen::Quaterniond r1 = {0.98877108, 0.0, 0.0, 0.14943813};
    Eigen::Quaterniond r2 = {0.98877108, 0.0, 0.0, -0.14943813};
    Eigen::Quaterniond r3 = {0.99875026, 0.0, 0.04997917, 0.0};
    Eigen::Quaterniond r4 = {0.99875026, 0.0, -0.04997917, 0.0};

    std::vector<Eigen::Quaterniond> r = {r1, r2, r3, r4};
    Eigen::Quaterniond avg = rotation_helpers::average_rotations(r);

    BOOST_REQUIRE(quat_are_close(avg, Eigen::Quaterniond::Identity()));
}

BOOST_AUTO_TEST_CASE(Sigma_Point_Conversions)
{
    // Generate a random initial state
    ukf::State initial_state;
    initial_state.position = Eigen::Vector3d::Random();
    initial_state.velocity = Eigen::Vector3d::Random();
    initial_state.orientation = {-0.325688076033, -0.783530090301, 0.201620620369, 0.489241249452};
    initial_state.angular_vel = Eigen::Vector3d::Random();
    initial_state.angular_acc = Eigen::Vector3d::Random();
    initial_state.acc_bias = Eigen::Vector3d::Random();
    initial_state.gyro_bias = Eigen::Vector3d::Random();

    // Random covariance
    ukf::Covariance initial_cov = ukf::Covariance::Identity().cwiseProduct(ukf::Covariance::Random().cwiseAbs());

    // Get sigma points from the function
    ukf::SigmaPoints sigmas = ukf_utils::compute_sigma_points(initial_state, initial_cov);

    // Let's hope the state we get out is the same
    ukf::StateAndCovariance output = ukf_utils::state_from_sigma_points(sigmas);

    BOOST_REQUIRE(states_are_close(output.state, initial_state));
    BOOST_REQUIRE(fabs(output.covariance.trace() - initial_cov.trace()) < A_TOL);

}
