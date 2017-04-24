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
// all the thread saftey
//
class SensorBase
{
public: // types //////////////////////////////////////////////////////////////
    //
    // type returned when data is requested
    //
    struct sensor_data_t
    {
        Timestamp timestamp;
        Eigen::MatrixXd measurement;
        Eigen::MatrixXd covariance;
    };

public: // destructor /////////////////////////////////////////////////////////
    //
    //
    //
    virtual ~SensorBase() {};

public: // methods ////////////////////////////////////////////////////////////
    //
    // when a new measurement comes in these functions can be called
    // TODO: Thread safety
    //
    inline void update_measurment(const Eigen::MatrixXd &data)
    {
        timestamp = Clock::now();
        measurement = data;
    }

    //
    // TODO: Thread safety
    //
    inline void update_measurment(const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariance_) 
    {
        timestamp = Clock::now();
        measurement = data;
        covariance = covariance_;
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
    // covariance associated with the measurment
    //
    Eigen::MatrixXd covariance;

}; // class SensorBase

//
// Base class to be inherited by from classes that represent inertial sensors
//
class InertialSensor : public SensorBase
{
public: // destructor /////////////////////////////////////////////////////////
    //
    //
    //
    virtual ~InertialSensor() {};

public: // Virtual Functions to Implemented ///////////////////////////////////
    //
    // Given an input state and a update dt, modify the state using data
    // held by the sensor.
    // Note: Some sensors may only implement one of these two functions
    //
    virtual void predict_state(State &state, double dt) = 0;

}; // class InertialSensorBase

//
// Base class to be inherited by from classes that represent observation sensors
//
class ObservationSensor : public SensorBase
{
public: // destructor /////////////////////////////////////////////////////////
    //
    //
    //
    virtual ~ObservationSensor() {};

public: // Virtual Functions to Implemented ///////////////////////////////////
    //
    // Given a state, use the current measurement data to transform the it
    //
    virtual void update_state(State &state) = 0;

    //
    // Convert the given state into an observation
    //
    virtual Eigen::MatrixXd convert_to_observation(State &state) = 0;

}; // class ObservationSesnor

///////////////////////////////////////////////////////////////////////////////
// sensor class definitions ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class AccelerometerInertial final: public InertialSensor
{
public: // constructor ///////////////////////////////////////////////////////
    //
    //
    //
    AccelerometerInertial() = default;

    //
    //
    //
    ~AccelerometerInertial() = default;

public: // methods ///////////////////////////////////////////////////////////

    void predict_state(State &state, double dt)
    {
        sensor_data_t sensor_data = get_sensor_data();
        std::cout << sensor_data.measurement.transpose();
        std::cout << " at " << std::chrono::system_clock::to_time_t(sensor_data.timestamp) << std::endl;
    }

}; // class AccelerometerInertial

} // namespace ukf
