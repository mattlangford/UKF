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
// These functions operate on rotation matricies
//
// Computes the mapping from R3 -> SO3
// Basically from the tangent space of one rotation to another rotation
//
Eigen::Matrix3d exp(const Eigen::Vector3d &w)
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
Eigen::Vector3d ln(const Eigen::Matrix3d &R)
{
    const double theta = acos((R.trace() - 1) / 2.0);
    const double factor = (0.5 + (theta * theta) / 12.0 + (7 * pow(theta, 4)) / 720.0);
    // std::cout << "theta " << theta << std::endl; 
    // std::cout << "factor " << factor << std::endl; 

    Eigen::Matrix3d out_mat = factor * (R - R.transpose());
    return {out_mat(2, 1), -out_mat(2, 0), out_mat(1, 0)};
}

