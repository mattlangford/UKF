#pragma once
#include <Eigen/Dense>

//
// Multiplication between two quaternions (which results in another quaternion)
// This is overloaded for any file that includes this header (that may change?)
//
Eigen::Quaterniond operator*(const Eigen::Quaterniond &rhs,
                             const Eigen::Quaterniond &lhs )
{
    const double w = rhs.w() * lhs.w() - rhs.x() * lhs.x() - rhs.y() * lhs.y() - rhs.z() * lhs.z();
    const double x = rhs.w() * lhs.x() + rhs.x() * lhs.w() + rhs.y() * lhs.z() - rhs.z() * lhs.y();
    const double y = rhs.w() * lhs.y() - rhs.x() * lhs.z() + rhs.y() * lhs.w() + rhs.z() * lhs.x();
    const double z = rhs.w() * lhs.z() + rhs.x() * lhs.y() - rhs.y() * lhs.x() + rhs.z() * lhs.w();

    return {w, x, y, z};
}

//
// Helper function to rotate a Vector by a Quaternion, the vector can be any magnitude
// and will be returned with the same magnitude it was passed in with
//
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

//
// These functions operate on rotation matrices
//
// Computes the mapping from R3 -> SO3
// Basically from the tangent space of one rotation to another rotation
//
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

//
// Goes the other way, SO3 -> R3 tangent space
//
Eigen::Vector3d ln_r(const Eigen::Matrix3d &R)
{
    const double theta = acos((R.trace() - 1) / 2.0);
    const double factor = (0.5 + (theta * theta) / 12.0 + (7 * pow(theta, 4)) / 720.0);

    Eigen::Matrix3d out_mat = factor * (R - R.transpose());
    return {out_mat(2, 1), -out_mat(2, 0), out_mat(1, 0)};
}

//
// These functions operate on Quaternions, same as the ones above
//
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

Eigen::Vector3d ln_q(const Eigen::Quaterniond &q)
{
    Eigen::Vector3d result;
    double theta = acos(q.w());

    if (theta < 1E-6)
    {
        return {0.0, 0.0, 0.0};
    }

    return 2 * q.vec() * theta / sin(theta);
}
