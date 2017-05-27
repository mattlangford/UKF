#pragma once
#include <Eigen/Dense>
#include "types.hh"

// UT Parameters //////////////////////////////////////////////////////////////
namespace params
{

//
// Normal UT parameters
//
constexpr double alpha = 1E-2;
constexpr double beta = 2;
constexpr double kappa = 0;
constexpr double lambda = alpha * alpha * (ukf::states::NUM_STATES + kappa) - ukf::states::NUM_STATES;

//
// Weights for computing sigma points, the first element should be the weight
// of the 0th sigma point and the other weights are applied to the rest
//
constexpr std::pair<double, double> mean_weights = {lambda / (ukf::states::NUM_STATES + lambda),
                                                    0.5 / ukf::states::NUM_STATES + lambda};
constexpr std::pair<double, double> cov_weights = {mean_weights.first + (1 - alpha * alpha + beta),
                                                   mean_weights.second};

//
// The factor multiplied by the square root of the covariance matrix before
// each UT, computed once here for convenience
//
const double sqrt_cov_factor = sqrt(ukf::states::NUM_STATES + lambda);

} // namespace params

// UKF Helper Functions ///////////////////////////////////////////////////////
namespace ukf_utils
{

//
// Convert a state and covariance into a set of discrete states as part of the
// Unscented Transform
//
ukf::SigmaPoints compute_sigma_points(const ukf::State& state,
                                      const ukf::Covariance& covariance);

//
// Given a set of sigma points, compute an expected State and associated
// covariance
//
ukf::StateAndCovariance state_from_sigma_points(const ukf::SigmaPoints &points);

} // namespace ukf_utils

// Operator Overloads /////////////////////////////////////////////////////////
//
// Multiplication between two quaternions (which results in another quaternion)
// This is overloaded for any file that includes this header (that may change?)
//
Eigen::Quaterniond operator*(const Eigen::Quaterniond &rhs,
                             const Eigen::Quaterniond &lhs);

// Rotation Helpers ///////////////////////////////////////////////////////////
namespace rotation_helpers
{
//
// Helper function to rotate a Vector by a Quaternion, the vector can be any magnitude
// and will be returned with the same magnitude it was passed in with
//
Eigen::Vector3d rotate_vec_by_quat(const Eigen::Vector3d & vector,
                                   const Eigen::Quaterniond &quat);

//
// Computes the mapping from R3 -> SO3
// Basically from the tangent space of one rotation to another rotation
//
Eigen::Matrix3d exp_r(const Eigen::Vector3d &w);

//
// Goes the other way, SO3 -> R3 tangent space
//
Eigen::Vector3d ln(const Eigen::Matrix3d &r);

//
// These functions operate on Quaternions, same as the ones above
//
Eigen::Quaterniond exp_q(const Eigen::Vector3d &w);

Eigen::Vector3d ln(const Eigen::Quaterniond &q);

inline Eigen::Matrix3d inverse(const Eigen::Matrix3d &rot);

inline Eigen::Quaterniond inverse(const Eigen::Quaterniond &rot);

//
// Compute the average of a set of rotation matrices
// http://ethaneade.com/lie.pdf pg.25
//
template <typename Rotation_t>
Rotation_t average_rotations(const std::vector<Rotation_t> &rots,
                             const size_t converge_iters = 3)
{
    Rotation_t u = rots[0];
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    //
    // At each iteration, find the distance between all rotations and the current
    // estimate for the mean. Update the mean to minimize the distance between all the
    // rotations
    //
    for (size_t iters = 0; iters < converge_iters; ++iters)
    {
        //
        // v keeps track of the distances (in the tangent space) between all the rotations
        //
        Eigen::MatrixXd v(3, rots.size());
        Rotation_t u_inv = inverse(u);
        for (size_t i = 0; i < rots.size(); ++i)
        {
            v.col(i) = ln(rots[i] * u_inv);
        }

        //
        // Update our rotation estimate for the mean using the mean computed in the
        // tangent space
        //
        Eigen::Vector3d tangent_space_avg = params::mean_weights.first * v.col(0);
        cov = params::cov_weights.first * v.col(0) * v.col(0).transpose();
        double cov_weight = params::cov_weights.second;
        double mean_weight = params::mean_weights.second;
        for (size_t i = 1; i < rots.size(); ++i)
        {
            cov += cov_weight * v.col(i) * v.col(i).transpose();
            tangent_space_avg += mean_weight * v.col(i);
        }
        u = exp_r(tangent_space_avg) * u;
    }

    return u;
}
} // namespace rotation_helpers

