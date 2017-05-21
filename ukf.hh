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
            ObsCovCrossCov obs = sensor->compute_observation(predicted_sigma_pts, predicted_state, params);

            //
            // Let's compute the error in our actual observed measurement and our predicted
            // observed measurement
            // Dims: 1 x OBS_SPACE_DIM
            //
            Eigen::MatrixXd observation_innovation = sensor->compute_innovation(latest_data.measurement,
                                                                                obs.observed_state);

            //
            // Add the sensor covariance in there
            // Dims: OBS_SPACE_DIM x OBS_SPACE_DIM
            //
            obs.covariance += latest_data.covariance;

            //
            // Compute kalman gain: Pxy * Pyy^-1
            // Dims: STATE_SPACE_DIM x OBS_SPACE_DIM
            //
            Eigen::MatrixXd kalman_gain = obs.cross_covariance * obs.covariance.inverse();

            //
            // Now time to update the actual state and covariance
            //
            predicted_state = predicted_state + kalman_gain * observation_innovation;
            predicted_cov = predicted_cov - kalman_gain * obs.covariance * kalman_gain.transpose();
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

        // TODO: Orientation from angular velocity, also check if this is correct
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

    //
    // Given a set of sigma points, compute an expected State and associated
    // covariance
    //
    StateAndCovariance state_from_sigma_points(const SigmaPoints &points)
    {
        //
        // TODO: Redo the rotation stuff using more operator overloading to make it cleaner
        //
        std::vector<Eigen::Matrix3d> rotations;
        rotations.reserve(NUM_SIGMA_POINTS);
        for(const State& sigma : points)
        {
            rotations.push_back(sigma.orientation);
        }

        // Compute mean
        double mean_weight = params.mean_weight.first;
        State new_state;
        for (const State& sigma_point : points)
        {
            new_state.position += mean_weight * sigma_point.position;
            new_state.velocity += mean_weight * sigma_point.velocity;
            new_state.acceleration += mean_weight * sigma_point.acceleration;
            new_state.angular_vel += mean_weight * sigma_point.angular_vel;
            new_state.angular_acc += mean_weight * sigma_point.angular_acc;
            new_state.acc_bias += mean_weight * sigma_point.acc_bias;
            new_state.gyro_bias += mean_weight * sigma_point.gyro_bias;

            mean_weight = params.mean_weight.second;
        }
        Eigen::Vector3d lie_mean = average_rotations(rotations);
        new_state.orientation = exp_r(lie_mean);

        // Compute covariance
        double cov_weight = params.cov_weight.first;
        Covariance new_covariance = Covariance::Zero();
        for (const State& sigma_point : points)
        {
            Eigen::Matrix<double, NUM_STATES, 1> innovation = sigma_point - new_state;
            new_covariance += cov_weight * innovation * innovation.transpose();
            cov_weight = params.cov_weight.second;
        }

        StateAndCovariance s_c;
        s_c.state = new_state;
        s_c.covariance = new_covariance;

        return s_c;

    }

    //
    // Compute the average of a set of rotation matrices
    //
    Eigen::Vector3d average_rotations(std::vector<Eigen::Matrix3d> rots) const
    {
        Eigen::Matrix3d u = rots[0];
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

        for (size_t iters = 0; iters < 5; ++iters)
        {
            Eigen::MatrixXd v(3, rots.size());
            Eigen::Matrix3d u_inv = u.inverse();
            cov = Eigen::Matrix3d::Zero();

            //
            // Build the v matrix
            //
            for (size_t i = 0; i < rots.size(); ++i)
            {
                v.col(i) = ln_r(rots[i] * u_inv);
            }

            //
            // Compute covariance and mean
            //
            Eigen::Vector3d tangent_space_avg = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < rots.size(); ++i)
            {
                double cov_weight = params.cov_weight.second;
                if (i == 0)
                    cov_weight = params.cov_weight.first;

                double mean_weight = params.mean_weight.second;
                if (i == 0)
                    cov_weight = params.mean_weight.first;

                cov += cov_weight * v.col(i) * v.col(i).transpose();
                tangent_space_avg += mean_weight * v.col(i);
            }
            u = exp_r(tangent_space_avg) * u;
        }
        return ln_r(u);
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
