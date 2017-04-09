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
