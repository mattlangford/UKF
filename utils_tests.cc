#include <Eigen/Dense>
#include <iostream>
#include <assert.h>

#include "utils.hh"

//
// File used for testing the utility functions
// TODO: Make into actual unti tests
//

bool is_close(const Eigen::Vector3d & a, const Eigen::Vector3d & b, double atol=1E-4)
{
    return (a - b).norm() <= atol;
}

void is_close_test()
{
    // Make sure the stupid is_close function actually works
    Eigen::Vector3d a(0, 1.0, 0.3);
    Eigen::Vector3d a_close(1E-5, 1.0, 0.3 + 1E-6);
    Eigen::Vector3d a_not_close(1.0, -1.0, 0.3);

    assert(is_close(a, a_close));
    assert(!is_close(a, a_not_close));

    std::cout << "PASS: is_close looks good." << std::endl;
}

void rotate_vec_by_quat_tests()
{
    // I used numpy and tf.transformations to confirm these
    Eigen::Vector3d forward(1, 0, 0);
    Eigen::Quaterniond no_rotation(1, 0, 0, 0);

    assert(is_close(forward, rotate_vec_by_quat(forward, no_rotation)));

    Eigen::Quaterniond x_90_z_90(0.5, 0.5, -0.5, 0.5);
    assert(is_close({0, 0, 1}, rotate_vec_by_quat(forward, x_90_z_90)));

    Eigen::Quaterniond random_1(-0.431410340004, -0.227475712562, -0.620743435549, 0.613854629334); 
    Eigen::Vector3d expected_1(-0.52427984, -0.24723836, -0.81486431);
    assert(is_close(expected_1, rotate_vec_by_quat(forward, random_1)));

    // Testing non normalized vectors
    Eigen::Vector3d random_vector(-0.60508391,  0.69961366,  1.22848347);
    Eigen::Quaterniond random_2(-0.325688076033, -0.783530090301, 0.201620620369, 0.489241249452);
    Eigen::Vector3d expected_2(-1.36749835, -0.49493513,  0.49974487);

    assert(is_close(expected_2, rotate_vec_by_quat(random_vector, random_2)));

    std::cout << "PASS: rotate_vec_by_quat (and by quaternion multiplicaion) look good." << std::endl;
}

int main()
{
    is_close_test();
    rotate_vec_by_quat_tests();

}
