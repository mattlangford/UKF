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
    std::vector<Eigen::Matrix3d> rots(4);
    rots[0] = exp({0.1, 0.1, 0.3});
    rots[1] = exp({0.1, -0.1, 0.0});
    rots[2] = exp({-0.1, 0.1, 0.0});
    rots[3] = exp({-0.1, -0.1, 0.3});

    Eigen::Matrix3d u = rots[0];
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    
    for (size_t iters = 0; iters < 5; ++iters)
    {
        Eigen::MatrixXd v(3, rots.size());
        Eigen::Matrix3d u_inv = u.inverse();
        // build the v matrix
        for (size_t i = 0; i < rots.size(); ++i)
        {
            v.col(i) = ln(rots[i] * u_inv);
        }

        // compute covariance and mean
        Eigen::Vector3d tangent_space_avg = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < rots.size(); ++i)
        {
            cov += (1.0 / rots.size()) * v.col(i) * v.col(i).transpose();
            tangent_space_avg += (1.0 / rots.size()) * v.col(i);
        }
        u = exp(tangent_space_avg) * u; 
    }
    std::cout << "=== RESULTS =============================" << std::endl;
    std::cout << u << std::endl << std::endl;
    std::cout << cov << std::endl;
}
