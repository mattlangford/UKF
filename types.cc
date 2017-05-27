#include "types.hh"
#include "utils.hh"

namespace ukf
{

State::State()
{
    position = Eigen::Vector3d::Zero();
    velocity = Eigen::Vector3d::Zero();
    acceleration = Eigen::Vector3d::Zero();
    orientation = Eigen::Quaterniond::Identity();
    angular_vel = Eigen::Vector3d::Zero();
    angular_acc = Eigen::Vector3d::Zero();
    acc_bias = Eigen::Vector3d::Zero();
    gyro_bias = Eigen::Vector3d::Zero();
}

//
// makes the sigma point transformation pretty easy
//
State State::operator+(const Eigen::Matrix<double, states::NUM_STATES, 1>& rhs) const
{
    State new_state;
    Eigen::Vector3d lie_rotation(rotation_helpers::ln(orientation));

    new_state.position = position + rhs.block<3, 1>(states::X, 0);
    new_state.velocity = velocity + rhs.block<3, 1>(states::VX, 0);
    new_state.acceleration = acceleration + rhs.block<3, 1>(states::AX, 0);
    new_state.orientation = rotation_helpers::exp_q(lie_rotation + rhs.block<3, 1>(states::RX, 0));
    new_state.angular_vel = angular_vel + rhs.block<3, 1>(states::WX, 0);
    new_state.angular_acc = angular_acc + rhs.block<3, 1>(states::aX, 0);
    new_state.acc_bias = acc_bias + rhs.block<3, 1>(states::AX_b, 0);
    new_state.gyro_bias = gyro_bias + rhs.block<3, 1>(states::GX_b, 0);

    return new_state;
};

//
// makes innovation computation easy
//
Eigen::Matrix<double, states::NUM_STATES, 1> State::operator-(const State &rhs) const
{
    Eigen::Matrix<double, states::NUM_STATES, 1> state_vector;
    Eigen::Vector3d lie_rotation(rotation_helpers::ln(orientation));
    Eigen::Vector3d rhs_lie_rotation(rotation_helpers::ln(rhs.orientation));

    state_vector.block<3, 1>(states::X, 0) = position - rhs.position;
    state_vector.block<3, 1>(states::VX, 0) = velocity - rhs.velocity;
    state_vector.block<3, 1>(states::AX, 0) = acceleration - rhs.acceleration;
    state_vector.block<3, 1>(states::RX, 0) = lie_rotation - rhs_lie_rotation;
    state_vector.block<3, 1>(states::WX, 0) = angular_vel - rhs.angular_vel;
    state_vector.block<3, 1>(states::aX, 0) = angular_acc - rhs.angular_acc;
    state_vector.block<3, 1>(states::AX_b, 0) = acc_bias - rhs.acc_bias;
    state_vector.block<3, 1>(states::GX_b, 0) = gyro_bias - rhs.gyro_bias;

    return state_vector;
};

void State::print() const
{
    std::cout << "position:     " << position.transpose() << std::endl;
    std::cout << "velocity:     " << velocity.transpose() << std::endl;
    std::cout << "acceleration: " << acceleration.transpose() << std::endl;
    std::cout << "orientation:  " << orientation << std::endl;
    std::cout << "angular_vel:  " << angular_vel.transpose()<< std::endl;
    std::cout << "angular_acc:  " << angular_acc.transpose()<< std::endl;
    std::cout << "acc_bias:     " << acc_bias.transpose() << std::endl;
    std::cout << "gyro_bias:    " << gyro_bias.transpose() << std::endl;
}

} // namespace
