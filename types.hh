#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <memory>

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
    Eigen::Quaterniond orientation;  // Quaternion
    Eigen::Vector3d    angular_vel;  // rads per second
    Eigen::Vector3d    angular_acc;  // rads per second per second
    Eigen::Vector3d    acc_bias;  // meters per second^2
    Eigen::Vector3d    gyro_bias;  // rads per second

    State();

    //
    // makes the sigma point transformation pretty easy
    //
    State operator+(const Eigen::Matrix<double, states::NUM_STATES, 1>& rhs) const;

    //
    // makes innovation computation easy
    //
    Eigen::Matrix<double, states::NUM_STATES, 1> operator-(const State &rhs) const;

    //
    //
    //
    void print() const;
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
// Struct used by sensors in the UKF for the measurement update step.
// All fields here are in the observation space (except cross_covariance)
//
struct ObsCovCrossCov
{
    Eigen::MatrixXd mean; // OBS_DIMS x 1
    Eigen::MatrixXd covariance; // OBS_DIMS x OBS_DIMS
    Eigen::MatrixXd cross_covariance; // STATE_DIMS x OBS_DIMS
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
