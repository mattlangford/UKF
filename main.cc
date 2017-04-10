#include <iostream>
#include <Eigen/Dense>


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

Eigen::Vector3d ln(const Eigen::Matrix3d &R)
{
    std::cout << ((R.trace() - 1) / 2.0) << std::endl;
    const double theta = acos((R.trace() - 1) / 2.0);
    const double factor = (0.5 + (theta * theta) / 12.0 + (7 * pow(theta, 4)) / 720.0);
    std::cout << "theta " << theta << std::endl;
    std::cout << "factor " << factor << std::endl;

    Eigen::Matrix3d out_mat = factor * (R - R.transpose());
    return {out_mat(2, 1), -out_mat(2, 0), out_mat(1, 0)};
}

int main()
{
    Eigen::Vector3d w(0, 0.1, 0.1);
    const Eigen::Matrix3d m = exp(w);
    std::cout << m << std::endl;
    std::cout << ln(m) << std::endl;
}
