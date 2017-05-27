#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

#include "utils.hh"
#include "types.hh"
#include "sensors.hh"

// TODO: Move function implementations over to the cc file

namespace ukf
{
//
// The main event
//
class UKF
{
public: // constructor /////////////////////////////////////////////////////////

    //
    // Construct with an initial state, a starting covariance for that state, a set
    // of inertial sensors, a list of observation sensors and some parameters used 
    // by the filter
    //
    UKF(const Sensors &sensors_, const ukf_params_t &params_) :
        sensors(sensors_),
        params(params_),
        last_time(Clock::now())
    {
    };

public: // methods /////////////////////////////////////////////////////////////

    //
    // Main function for the ukf, call this and the filter will use the most
    // recent measurements from all the given sensors to compute an estimated
    // State and Covariance for you
    //
    StateAndCovariance update(const StateAndCovariance in, const Timestamp &time)
    {
        //
        // Compute sigmas points in the state space, they aren't really predicted sigma points
        // yet, but they will be shortly
        //
        SigmaPoints predicted_sigma_pts = compute_sigma_points(in.state, in.covariance);

        //
        // First do the time update
        // TODO: Ideally, we'd do a time update between each sensor measurement in the loop below
        //
        const Timestamp update_time = Clock::now();
        for(State &point : predicted_sigma_pts)
        {
            time_update_f(point, update_time);
        }

        //
        // Save the update time for the next run
        //
        last_time = update_time;

        //
        // We need to compute an initial predicted state which will be corrected
        // with each additional sensor measurement we have
        //
        StateAndCovariance predicted = state_from_sigma_points(predicted_sigma_pts);
        State predicted_state = predicted.state;
        Covariance predicted_cov = predicted.covariance + compute_model_covariance();

        //
        // Loop through each sensor and let it update the state how it wants
        // TODO: This should be ordered based on the incoming time of each sensor
        //
        for (SensorPtr sensor : sensors)
        {
            //
            // Fetch the latest sensor data so that any sensor updates don't break things
            // halfway into the update
            //
            sensor_data_t latest_data = sensor->get_sensor_data();

            //
            // Convert our predicted states in the state space into a predicted observation
            // (and covariance) in the observation space. Also compute the cross covariance
            // for the predicted states and the observations
            //
            ObsCovCrossCov observation_data =
                sensor->compute_observation(predicted_sigma_pts, predicted_state, params);

            //
            // Let's compute the error in our actual observed measurement and our predicted
            // observed measurement
            // Dims: 1 x OBS_SPACE_DIM
            //
            Eigen::MatrixXd observation_innovation = sensor->compute_innovation(latest_data.measurement,
                                                                                observation_data.mean);

            //
            // Add the sensor covariance in there
            // Dims: OBS_SPACE_DIM x OBS_SPACE_DIM
            //
            observation_data.covariance += latest_data.covariance;

            //
            // Compute kalman gain: Pxy * Pyy^-1
            // Dims: STATE_SPACE_DIM x OBS_SPACE_DIM
            //
            Eigen::MatrixXd kalman_gain = observation_data.cross_covariance * observation_data.covariance.inverse();

            //
            // Now time to update the actual state and covariance
            //
            predicted_state = predicted_state + kalman_gain * observation_innovation;
            predicted_cov = predicted_cov - kalman_gain * observation_data.covariance * kalman_gain.transpose();
            SigmaPoints predicted_sigma_pts = compute_sigma_points(predicted_state, predicted_cov);
        }

        //
        // Return to the user the final state and covariance from the above loop
        //
        StateAndCovariance output;
        output.state = predicted_state;
        output.covariance = predicted_cov;
        return output;
    }

// temporarily public for testing
public: // methods ////////////////////////////////////////////////////////////

    //
    // Update our current estimate of the state according to the update time
    // This is assuming constant accelerations
    // This function is passed to the UT during the time update
    //
    void time_update_f(State& current_state, const Timestamp &time) const
    {
        std::chrono::duration<double> dt_ = time - last_time;
        const double dt = 1.0;// dt_.count();

        current_state.position += current_state.velocity * dt + 0.5 * current_state.acceleration * dt * dt;
        current_state.velocity += current_state.acceleration * dt;

        current_state.orientation = exp_q(current_state.angular_vel) * current_state.orientation;
        current_state.angular_vel += current_state.angular_acc * dt;
    }

    //
    // Helper function that computes the model covariance. This is probably fine to hard code
    // like this, since it will likely not change
    // TODO: Ensure this is a sensible model covariance. All diagonal values may need to not 
    // be the same...
    //
    Covariance compute_model_covariance() const
    {
        return Covariance::Identity() * 1E-3;
    }

    //
    // Compute sigma points from a given StateAndCovariance
    //
    SigmaPoints compute_sigma_points(const State &state,
                                     const Covariance& covariance) const
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
            std::cout << state.velocity.transpose() << std::endl;
        }

        return sigmas;
    }

private: // members ////////////////////////////////////////////////////////////
    //
    // List of the sensors in our filter.
    // This can go in the update function call
    //
    Sensors sensors;

    //
    // Some precomputed parameters for the UT
    //
    ukf_params_t params;

    //
    // Used to compute dt for updates
    //
    Timestamp last_time;

};

}  // namespace
