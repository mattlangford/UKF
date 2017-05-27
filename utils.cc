#include "utils.hh"

// UT Parameters //////////////////////////////////////////////////////////////
Eigen::Quaterniond operator*(const Eigen::Quaterniond &rhs,
                             const Eigen::Quaterniond &lhs)
{
    const double w = rhs.w() * lhs.w() - rhs.x() * lhs.x() - rhs.y() * lhs.y() - rhs.z() * lhs.z();
    const double x = rhs.w() * lhs.x() + rhs.x() * lhs.w() + rhs.y() * lhs.z() - rhs.z() * lhs.y();
    const double y = rhs.w() * lhs.y() - rhs.x() * lhs.z() + rhs.y() * lhs.w() + rhs.z() * lhs.x();
    const double z = rhs.w() * lhs.z() + rhs.x() * lhs.y() - rhs.y() * lhs.x() + rhs.z() * lhs.w();

    return {w, x, y, z};
}

// UKF Helper Functions ///////////////////////////////////////////////////////
namespace ukf_utils
{

ukf::SigmaPoints compute_sigma_points(const ukf::State& state,
                                      const ukf::Covariance& covariance)
{
    //
    // Compute matrix square root, this isn't really a Covariance, but
    // the type is convenient to use
    //
    ukf::Covariance cov_sqrt = covariance.llt().matrixL();
    cov_sqrt *= params::sqrt_cov_factor;

    //
    // Populate sigma vector, we fill out sigmas[0] with
    // the original state.
    // NOTE: The + operator is overloaded for the State type so that this'll work
    //
    ukf::SigmaPoints sigmas(ukf::NUM_SIGMA_POINTS);
    sigmas[0] = state;
    for (size_t i = 1; i < ukf::NUM_STATES; ++i)
    {
        sigmas[2 * i] = state + cov_sqrt.row(i);
        sigmas[2 * i + 1] = state + -cov_sqrt.row(i);
    }

    return sigmas;
}

ukf::StateAndCovariance state_from_sigma_points(const ukf::SigmaPoints &points)
{
    //
    // Compute mean
    //
    double mean_weight = params::mean_weights.first;
    ukf::State new_state;
    for (const ukf::State& sigma_point : points)
    {
        new_state.position += mean_weight * sigma_point.position;
        new_state.velocity += mean_weight * sigma_point.velocity;
        new_state.acceleration += mean_weight * sigma_point.acceleration;
        new_state.angular_vel += mean_weight * sigma_point.angular_vel;
        new_state.angular_acc += mean_weight * sigma_point.angular_acc;
        new_state.acc_bias += mean_weight * sigma_point.acc_bias;
        new_state.gyro_bias += mean_weight * sigma_point.gyro_bias;

        mean_weight = params::mean_weights.second;
    }

    //
    // TODO: Redo the rotation stuff using more operator overloading to make it cleaner
    //
    std::vector<Eigen::Quaterniond> rotations;
    rotations.reserve(ukf::NUM_SIGMA_POINTS);
    for(const ukf::State& sigma : points)
    {
        rotations.push_back(sigma.orientation);
    }
    new_state.orientation = rotation_helpers::average_rotations(rotations);

    // Compute covariance
    double cov_weight = params::cov_weights.first;
    ukf::Covariance new_covariance = ukf::Covariance::Zero();
    for (const ukf::State& sigma_point : points)
    {
        Eigen::Matrix<double, ukf::NUM_STATES, 1> innovation = sigma_point - new_state;
        new_covariance += cov_weight * innovation * innovation.transpose();
        cov_weight = params::cov_weights.second;
    }

    ukf::StateAndCovariance s_c;
    s_c.state = new_state;
    s_c.covariance = new_covariance;

    return s_c;

}
} // namespace ukf_utils

// Rotation Helpers ///////////////////////////////////////////////////////////
namespace rotation_helpers
{
Eigen::Vector3d rotate_vec_by_quat(const Eigen::Vector3d & vector,
                                   const Eigen::Quaterniond &quat)
{
    const Eigen::Quaterniond quat_conj(quat.w(), -quat.x(), -quat.y(), -quat.z());
    const Eigen::Quaterniond vector_quat(0, vector.x(), vector.y(), vector.z());

    // Need to normalize vector_quat since it's not valid if it's not
    const Eigen::Quaterniond res = quat * vector_quat.normalized() * quat_conj;

    // The caller will want a vector of the same magnitude that they gave us
    return vector_quat.norm() * Eigen::Vector3d(res.x(), res.y(), res.z());
}

Eigen::Matrix3d exp_r(const Eigen::Vector3d &w)
{
    // skew symmetric matrix forming the Lie Algebra
    Eigen::Matrix3d w_x;
    w_x <<   0.0, -w.z(),  w.y(),
           w.z(),    0.0, -w.x(),
          -w.y(),  w.x(),    0.0;

    // compute theta squared which can be used without square rooting
    const double theta_2 = w.transpose() * w;

    // taylor series expansion
    Eigen::Matrix3d second_term = w_x;
    second_term *= (1 - theta_2 / 6.0 + (theta_2 * theta_2) / 120.0);

    Eigen::Matrix3d third_term = w_x * w_x;
    third_term *= (0.5 - theta_2 / 12.0 + (theta_2 * theta_2) / 720.0);

    return Eigen::Matrix3d::Identity(3, 3) + second_term + third_term;
}

Eigen::Vector3d ln(const Eigen::Matrix3d &R)
{
    const double theta = acos((R.trace() - 1) / 2.0);
    const double factor = (0.5 + (theta * theta) / 12.0 + (7 * pow(theta, 4)) / 720.0);

    Eigen::Matrix3d out_mat = factor * (R - R.transpose());
    return {out_mat(2, 1), -out_mat(2, 0), out_mat(1, 0)};
}

Eigen::Quaterniond exp_q(const Eigen::Vector3d &w)
{
    double theta = w.norm();
    if (theta < 1E-6)
    {
        return {1.0, 0.0, 0.0, 0.0};
    }

    Eigen::Quaterniond result;
    result.w() = cos(theta / 2.0);
    result.vec() = w * sin(theta / 2.0) / theta;
    return result;
}

Eigen::Vector3d ln(const Eigen::Quaterniond &q)
{
    Eigen::Vector3d result;
    double theta = acos(q.w());

    if (theta < 1E-6)
    {
        return {0.0, 0.0, 0.0};
    }

    return 2 * q.vec() * theta / sin(theta);
}

inline Eigen::Matrix3d inverse(const Eigen::Matrix3d &rot)
{
    return rot.transpose();
}

inline Eigen::Quaterniond inverse(const Eigen::Quaterniond &rot)
{
    return {rot.w(), -rot.x(), -rot.y(), -rot.z()};
}
} // namespace rotation_helpers

