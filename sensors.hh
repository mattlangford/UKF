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
public: // Virtual Functions to Implemented ///////////////////////////////////
    //
    // Given a set of predicted sigma points, return to the caller a predicted
    // observation and covariance associated with that in the observation space
    // along with a cross covariance between the state and observation space
    //
    virtual ObsCovCrossCov compute_observation(const SigmaPoints &points) const = 0;

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
    ObsCovCrossCov compute_observation(const SigmaPoints &points) const
    {
    };

    //
    // Needed for the kalman update, computes the error in two observations
    //
    Eigen::MatrixXd compute_innovation(const Eigen::MatrixXd &actual_observation,
                                       const Eigen::MatrixXd &predicted_observation) const
    {

    };

}; // class Accelerometer

} // namespace ukf
