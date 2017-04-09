#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd transform(Eigen::MatrixXd points)
{
    Eigen::Vector2d t_vec(10, -5);
    points = points.colwise() + t_vec;

    return points;
}

int main()
{
    double dims = 2;
    double alpha = 1E-3;
    double beta = 2;
    double kappa = 0;
    double lambda = alpha * alpha * (dims + kappa) - dims;
    std::cout << lambda << std::endl;

    Eigen::Matrix2d mat;
    
    mat << 1.0, 0.5, 
           0.3, 1.0;

    const Eigen::LLT<Eigen::Matrix2d> llt = mat.llt();
    Eigen::Matrix2d sqrt_m = llt.matrixL();
    sqrt_m *= sqrtf(dims + lambda);

    const Eigen::Vector2d mean(0.3, 100);
    Eigen::Matrix<double, 2, 5> sigmas = Eigen::MatrixXd::Zero(2, 5);

    sigmas.block<2, 1>(0, 0) = mean;
    sigmas.block<2, 2>(0, 1) = (sqrt_m).colwise() + mean;
    sigmas.block<2, 2>(0, 3) = (-sqrt_m).colwise() + mean;

    sigmas = transform(sigmas);

    double wm_0 = (lambda / (dims + lambda));
    double wm_i = 0.5 / (dims + lambda);

    Eigen::Vector2d new_mean = Eigen::Vector2d::Zero(2, 1);
    for (size_t i = 0; i < 5; ++i)
    {
        double weight = i == 0 ? wm_0 : wm_i;
        new_mean += weight * sigmas.col(i);
    }

    Eigen::MatrixXd mean_centered = sigmas.colwise() - new_mean;

    // Recompute covariance
    double wc_0 = (lambda / (dims + lambda)) + (1 - alpha * alpha + beta);
    double wc_i = 0.5 / (dims + lambda);

    std::cout << wc_0 << " " << wc_i << std::endl;
    
    Eigen::Matrix2d new_covariance = Eigen::Matrix2d::Zero(2, 2);
    for (size_t i = 0; i < 5; ++i)
    {
        double weight = i == 0 ? wc_0 : wc_i;
        std::cout << weight << std::endl;
        std::cout << mean_centered.col(i) * mean_centered.col(i).transpose() << std::endl;
        new_covariance += weight * mean_centered.col(i) * mean_centered.col(i).transpose();
    }

    std::cout << "new_mean" << std::endl;
    std::cout << new_mean << std::endl;
    std::cout << "new_covariance" << std::endl;
    std::cout << new_covariance << std::endl;
    std::cout << std::endl;

    std::cout << sigmas << std::endl;
}
