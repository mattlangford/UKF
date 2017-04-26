#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <memory>

#include "utils.hh"

// TODO: Move function implementations over to the cc file

namespace ukf
{

//
// Some basic typedefs not really related to the filter
//
using Clock     = std::chrono::high_resolution_clock;
using Timestamp = Clock::time_point;

//
// enum for indexing covariance matrix, these don't really apply
// to the main State type since it's a struct
//
enum states : uint8_t
{
    X,
    Y,
    Z,
    VX,
    VY,
    VZ,
    AX,
    AY,
    AZ,
    RX,
    RY,
    RZ,
    WX,
    WY,
    WZ,
    aX,
    aY,
    aZ,
    AX_b,  // meas_a = R * (real_a - bias)
    AY_b,
    AZ_b,
    GX_b,  // TODO
    GY_b,
    GZ_b,

    NUM_STATES
};

//
// List of sensors
//   accelerometer -> acceleration | roll | pitch
//   camera        -> velocity | angular velocity
//   gyro          -> angular velocity
//   magnetometer  -> yaw
//

//
// Plan for acc bias:
//   measured_a = R * (real_a - bias)
//   // assuming real_a = 1g
//   measured_a = R * ([0, 0, 1] - bias)
//   bias = -R_inv * measured_a + [0, 0, 1]
//

// This is an alternate way to keep track of the state...
struct State
{
    Eigen::Vector3d    position;  // meters
    Eigen::Vector3d    velocity;  // meters per second
    Eigen::Vector3d    acceleration;  // meters per second per second
    Eigen::Matrix3d    orientation;  // temporarily a rotation matrix
    Eigen::Vector3d    angular_vel;  // rads per second
    Eigen::Vector3d    angular_acc;  // rads per second per second
    Eigen::Vector3d    acc_bias;  // meters per second^2
    Eigen::Vector3d    gyro_bias;  // rads per second

    State()
    {
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        acceleration = Eigen::Vector3d::Zero();
        orientation = Eigen::Matrix3d::Identity();
        angular_vel = Eigen::Vector3d::Zero();
        angular_acc = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        gyro_bias = Eigen::Vector3d::Zero();
    }

    // makes the sigma point transformation pretty easy
    State operator+(const Eigen::Matrix<double, states::NUM_STATES, 1>& lhs) const
    {
        State new_state;
        Eigen::Vector3d lie_rotation(ln(orientation));
        new_state.position = position + lhs.block<3, 1>(states::X, 0);
        new_state.velocity = velocity + lhs.block<3, 1>(states::VX, 0);
        new_state.velocity = acceleration + lhs.block<3, 1>(states::AX, 0);
        new_state.orientation = exp(lie_rotation + lhs.block<3, 1>(states::RX, 0));
        new_state.angular_vel = angular_vel + lhs.block<3, 1>(states::WX, 0);
        new_state.angular_acc = angular_acc + lhs.block<3, 1>(states::aX, 0);
        new_state.acc_bias = acc_bias + lhs.block<3, 1>(states::AX_b, 0);
        new_state.gyro_bias = gyro_bias + lhs.block<3, 1>(states::GX_b, 0);

        return new_state;
    };

    void print() const
    {
        std::cout << "position:           " << position.transpose() << std::endl;
        std::cout << "velocity:           " << velocity.transpose() << std::endl;
        std::cout << "acceleration:       " << acceleration.transpose() << std::endl;
        std::cout << "orientation (x vec) " << orientation.block<3, 1>(0, 0).transpose()<< std::endl;
        std::cout << "angular_vel:        " << angular_vel.transpose()<< std::endl;
        std::cout << "angular_acc:        " << angular_acc.transpose()<< std::endl;
        std::cout << "acc_bias:           " << acc_bias.transpose() << std::endl;
        std::cout << "gyro_bias:          " << gyro_bias.transpose() << std::endl;
    }
};

//
// Covariance matrix, use the states enum to index this
//
using Covariance = Eigen::Matrix<double, states::NUM_STATES, states::NUM_STATES>;

//
// Structs returned from and used by the UKF
//
struct StateAndCovariance
{
    State      state;
    Covariance covariance;

    void print() const
    {
        std::cout << "State: " << std::endl;
        state.print();
        std::cout << "Covariance: " << std::endl;
        std::cout << covariance << std::endl;
    }
};

//
// Struct used by sensors in the UKF for the measurement update step
//
struct ObsCovCrossCov
{
    Eigen::MatrixXd   observed_state;
    Eigen::MatrixXd   covariance;
    Eigen::MatrixXd   cross_covariance;
};

//
// Computed during the UT, a list of 2 * states::NUM_STATES + 1 State's that
// are passed into the transition function for each UT. The transition function
// should return a new set of SigmaPoints.
//
using SigmaPoints = std::vector<State>;

//
// There should be 2*n + 1 sigma points, we can pre-compute it here
//
constexpr size_t NUM_SIGMA_POINTS = 2 * states::NUM_STATES + 1;

//
// Observational sensors are used using during the measurement update, forward declare that here
// and typedef a vector of them for passing around. There may be a better name than this
//
class SensorBase;
using SensorPtr = std::shared_ptr<SensorBase>;
using Sensors = std::vector<SensorPtr>;

//
// Datum returned by each Sensor type
//
struct sensor_data_t
{
    Timestamp timestamp;
    Eigen::MatrixXd measurement;
    Eigen::MatrixXd covariance;
};

}  // namespace
