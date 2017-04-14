#include <iostream>
#include <Eigen/Dense>

#include "ukf.hh"


void average_rotations(std::vector<Eigen::Matrix3d> rots)
{
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

using namespace ukf;
int main()
{
    ukf_params_t params;    
    State initial_state;     
    Covariance initial_covariance = Covariance::Identity() * 1E-3;

    // Let's do the transform now (not really covariance type, but close enough)
    Covariance cov_sqrt = initial_covariance.llt().matrixL();
    cov_sqrt *= params.sqrt_cov_factor;
    std::cout << "cov_sqrt " << std::endl << cov_sqrt << std::endl;

    SigmaPoints sigmas(NUM_SIGMA_POINTS);
    sigmas[0] = initial_state;
    for (size_t i = 0; i < NUM_STATES; ++i)
    {
        sigmas[2 * i] = initial_state + cov_sqrt.row(i);
        sigmas[2 * i + 1] = initial_state + -cov_sqrt.row(i);
    }
    
    std::cout << sigmas[1].position << std::endl;
}
