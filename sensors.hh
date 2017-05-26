#pragma once
#include <Eigen/Dense>

#include "types.hh"

namespace ukf
{

///////////////////////////////////////////////////////////////////////////////
// base class definitions /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//
// Base class for all sensors, includes getters and setters for things and
// all the thread safety.
//
class SensorBase
{
public: // Destructor /////////////////////////////////////////////////////////
    //
    //
    //
    virtual ~SensorBase() {};

public: // Virtual Functions to Implemented ///////////////////////////////////
    //
    // Given a set of predicted sigma points, return to the caller a predicted
    // observation and covariance associated with that in the observation space
    // along with a cross covariance between the state and observation space
    //
    virtual ObsCovCrossCov compute_observation(const SigmaPoints &points,
                                               const State &predicted_state,
                                               const ukf_params_t &params) const = 0;

    //
    // Needed for the kalman update, computes the error in two observations
    //
    virtual Eigen::MatrixXd compute_innovation(const Eigen::MatrixXd &actual_observation,
                                               const Eigen::MatrixXd &predicted_observation) const = 0;

public: // methods ////////////////////////////////////////////////////////////
    //
    // when a new measurement comes in these functions can be called
    //
    inline void update_measurment(const Eigen::MatrixXd &data)
    {
        update_measurment(data, Clock::now());
    }

    //
    // A lot of times the covariance won't need to updated
    // TODO: Thread safety
    //
    inline void update_measurment(const Eigen::MatrixXd &data, const Timestamp &time)
    {
        timestamp = Clock::now();
        measurement = data;
    }

    //
    // In the cases you want to update with a new sensor covariance, this exists
    //
    inline void update_measurment(const Eigen::MatrixXd &data, const Eigen::MatrixXd &cov)
    {
        update_measurment(data, cov, Clock::now());
    }

    //
    // TODO: Thread safety
    //
    inline void update_measurment(const Eigen::MatrixXd &data, const Eigen::MatrixXd &cov, const Timestamp &time)
    {
        timestamp = time;
        measurement = data;
        covariance = cov;
    }


    //
    // Get the current data held by this sensor
    // TODO: Thread safety
    //
    inline sensor_data_t get_sensor_data() const
    {
        sensor_data_t return_data;
        return_data.timestamp = timestamp;
        return_data.measurement = measurement;
        return_data.covariance = covariance;
        return return_data;
    }

private: // private members ///////////////////////////////////////////////////
    //
    // When was the last measurement update
    //
    Timestamp timestamp;

    //
    // measurement vector updated each time a new sensor measurement comes in
    //
    Eigen::MatrixXd measurement;

    //
    // covariance associated with the measurement
    //
    Eigen::MatrixXd covariance;

}; // class SensorBase

///////////////////////////////////////////////////////////////////////////////
// sensor class definitions ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class Accelerometer final: public SensorBase
{

//
// The accelerometer uses it's measurements to update the AX, AY, and AZ states
// along with the biases associated with each of those.
// Eventually it will also play a roll in RX, and RY as well - soon.
//
// Below is an enum for milling around in the observation vector
//
enum obs_space : uint8_t
{
    ax,
    ay,
    az,

    OBS_SPACE_DIM
};

public: // constructor ///////////////////////////////////////////////////////
    //
    //
    //
    Accelerometer() = default;

    //
    //
    //
    ~Accelerometer() = default;

public: // methods ///////////////////////////////////////////////////////////
    //
    // Given a set of predicted sigma points, return to the caller a StateCovCrossCov type
    //
    ObsCovCrossCov compute_observation(const SigmaPoints &points,
                                       const State &predicted_state,
                                       const ukf_params_t &params) const
    {
        // Loop through each sigma point and compute the expected observation we'd expect
        // to see for that state
        Eigen::MatrixXd expected_observations = Eigen::MatrixXd::Zero(OBS_SPACE_DIM, points.size());
        for (size_t i = 0; i < points.size(); ++i)
        {
            const State &state = points[i];
            expected_observations.col(i) = state.acceleration;
        }

        //
        // Compute the output type, these three steps could be done together to be faster.
        // Also it is left to be as general as possible to allow more complex stuff
        //
        ObsCovCrossCov output;

        //
        // Compute the weighted mean of those measurements
        //
        output.mean = expected_observations.col(0) * params.mean_weight.first;
        double mean_weight = params.mean_weight.second;
        for (size_t i = 1; i < expected_observations.cols(); ++i)
        {
            const Eigen::Matrix<double, OBS_SPACE_DIM, 1> &observation = expected_observations.col(i);
            output.mean += mean_weight * observation;
        }

        //
        // Compute the weighted covariance of those measurements
        //
        double cov_weight = params.cov_weight.first;
        output.covariance = Eigen::Matrix<double, OBS_SPACE_DIM, OBS_SPACE_DIM>::Zero();
        for (size_t i = 0; i < expected_observations.cols(); ++i)
        {
            const Eigen::Matrix<double, OBS_SPACE_DIM, 1> &innovation =
                expected_observations.col(i) - output.mean;

            output.covariance += cov_weight * innovation * innovation.transpose();

            cov_weight = params.cov_weight.second;
        }

        //
        // Compute the weighted cross covariance of the states and measurements
        //
        cov_weight = params.cov_weight.first;
        output.cross_covariance = Eigen::Matrix<double, states::NUM_STATES, OBS_SPACE_DIM>::Zero();
        for (size_t i = 0; i < expected_observations.cols(); ++i)
        {
            const Eigen::Matrix<double, states::NUM_STATES, 1> &state_innovation =
                points[i] - predicted_state;

            const Eigen::Matrix<double, OBS_SPACE_DIM, 1> &obs_innovation =
                expected_observations.col(i) - output.mean;

            output.covariance += cov_weight * state_innovation * obs_innovation.transpose();

            cov_weight = params.cov_weight.second;
        }

        return output;
    };

    //
    // Needed for the kalman update, computes the error in two observations
    //
    Eigen::MatrixXd compute_innovation(const Eigen::MatrixXd &actual_observation,
                                       const Eigen::MatrixXd &predicted_observation) const
    {
        return actual_observation - predicted_observation;
    };

}; // class Accelerometer

} // namespace ukf
