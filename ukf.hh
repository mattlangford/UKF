#include <iostream>
#include <Eigen/Dense>
#include <chrono>

#include "utils.hh"

// TODO: Move function implementations over to the cc file

namespace ukf
{

//
// Some basic typedefs not really related to the filter
//
using Clock     = std::chrono::high_resolution_clock;
using Timestamp = Clock::time_point;

//
// enum for indexing covariance matrix, these don't really apply
// to the main State type since it's a struct
//
enum states : uint8_t
{
    X,
    Y,
    Z,
    VX,
    VY,
    VZ,
    RX,  
    RY,
    RZ,
    WX,
    WY,
    WZ,
    AX_b,  // meas_a = R * (real_a - bias)
    AY_b,
    AZ_b,
    GX_b,  // TODO
    GY_b,
    GZ_b,

    NUM_STATES
};

//
// List of sensors
//   accelerometer -> acceleration | roll | pitch
//   camera        -> velocity | angular velocity
//   gyro          -> angular velocity
//   magnetometer  -> yaw
//

//
// Plan for acc bias:
//   measured_a = R * (real_a - bias)
//   // assuming real_a = 1g
//   measured_a = R * ([0, 0, 1] - bias)
//   bias = -R_inv * measured_a + [0, 0, 1]
//

// This is an alternate way to keep track of the state...
struct State
{
    Eigen::Vector3d    position;  // meters
    Eigen::Vector3d    velocity;  // meters per second
    Eigen::Matrix3d    orientation;  // temporarily a rotation matrix
    Eigen::Vector3d    angular_vel;  // rads per second
    Eigen::Vector3d    acc_bias;  // meters per second^2
    Eigen::Vector3d    gyro_bias;  // rads per second

    State()
    {
        // everything should get auto generated expect maybe this
        orientation = Eigen::Matrix3d::Identity();
    }

    // makes the sigma point transformation pretty easy
    State operator+(const Eigen::Matrix<double, states::NUM_STATES, 1>& lhs)
    {
        State new_state;
        Eigen::Vector3d lie_rotation(ln(orientation));      
        std::cout << "lie_rotation " << lie_rotation << std::endl;
        new_state.position = position + lhs.block<3, 1>(states::X, 0);
        new_state.velocity = velocity + lhs.block<3, 1>(states::VX, 0);
        new_state.orientation = exp(lie_rotation + lhs.block<3, 1>(states::RX, 0));
        std::cout << "rot_rotation " << new_state.orientation << std::endl;
        new_state.angular_vel = angular_vel + lhs.block<3, 1>(states::WX, 0);
        new_state.acc_bias = acc_bias + lhs.block<3, 1>(states::AX_b, 0);
        new_state.gyro_bias = gyro_bias + lhs.block<3, 1>(states::GX_b, 0);

        return new_state;
    };
};

//
// Covariance matrix, use the states enum to index this
//
using Covariance = Eigen::Matrix<double, states::NUM_STATES, states::NUM_STATES>;
//
// Computed during the UT, a list of 2 * states::NUM_STATES + 1 State's that
// are passed into the transition function for each UT. The transition function
// should return a new set of SigmaPoints.
//
using SigmaPoints = std::vector<State>;

//
// There should be 2*n + 1 sigma points, we can precompute this here
//
constexpr size_t NUM_SIGMA_POINTS = 2 * states::NUM_STATES + 1;

//
// Structs returned from the UKF
//
struct StateAndCovariance
{
    State      state;
    Covariance covariance;
};
struct StateAndCrossCovariance
{
    State      state;
    Covariance cross_covariance;
};

//
// Parameters needed to do the UT, a lot of these are precalculated in order
// to save compute time later on
//
struct ukf_params_t
{
    // Could be better with some const
    ukf_params_t(double alpha_=1E-3, double beta_=2, double kappa_=0)
    {
        alpha = alpha_;
        beta = beta_;
        kappa = kappa_;
        lambda = alpha * alpha * (states::NUM_STATES + kappa) - states::NUM_STATES;
        
        mean_weight.first = lambda / (states::NUM_STATES + lambda);
        mean_weight.second = 0.5 / (states::NUM_STATES + lambda);

        cov_weight.first = mean_weight.first + (1 - alpha * alpha + beta);
        cov_weight.second = mean_weight.second;

        sqrt_cov_factor = sqrtf(states::NUM_STATES + lambda);
    };

    // normal parameters
    double alpha;
    double beta;
    double kappa;
    double lambda;    

    // weights for computing sigma points, the first element should be the weight
    // of the 0th sigma point and the other weights are applied to the rest
    std::pair<double, double> mean_weight;
    std::pair<double, double> cov_weight;

    // the factor multiplied by the square root of the covairance matrix before
    // each UT
    double sqrt_cov_factor;
};


//
// The main event
//
class UKF
{
public: // constructor /////////////////////////////////////////////////////////

    //
    // construt with an initial state and a starting covariance for that states
    // along with some parameters used by the filter
    //
    UKF(State initial_state, Covariance initial_covariance, ukf_params_t params_) :
        state(initial_state), 
        covariance(initial_covariance), 
        params(params_), 
        last_time(Clock::now())
    {};
    
public: // methods /////////////////////////////////////////////////////////////

    //
    // Called with a new accelerometer measurement and time. This will update
    // the estimated position, velocity, and rotation.
    //
    void got_accelerometer(const Eigen::Vector3d & body_acc_biased, 
                           const Eigen::Matrix3d & acc_cov, 
                           const Timestamp &       time)
    {
        const double dt = std::chrono::duration<double>(time - last_time).count();

        // First thing we will do is update positional states
        const auto position_velocity_update = 
            [&body_acc_biased, &dt](const SigmaPoints &in_states)
        {
            // Take the accelerometer numbers, and apply them to the position
            // and velocty states
            SigmaPoints out_states;
            out_states.reserve(NUM_SIGMA_POINTS);
            for (const State & state : in_states)
            {
                State new_state = std::move(state);
                const Eigen::Vector3d acc_body = body_acc_biased - new_state.acc_bias;
                const Eigen::Vector3d acc_world = body_acc_biased - new_state.acc_bias;

                new_state.velocity += acc_body;
            }

        };

        // Next we will attempt to correct our roll and pitch orientation
        // using the assumption gravity is the only present acceleration 
        

        // Update our bias

    }
    
private: // methods ////////////////////////////////////////////////////////////
    
    //
    //
    //
    template <typename Func>
    StateAndCovariance unscented_transform(const State &      initial_state, 
                                           const Covariance & initial_covariance,
                                           const Func         transition_f) const
    {
        // TODO: Initial tests looked good for R3, but this needs to work on SO3 and R3

        // LDLT is better I think
        const Eigen::LLT<Covariance> cov_sqrt(initial_covariance.llt().matrixL());


        // Recompute covariance
        // for (size_t i = 0; i < 5; ++i)
        // {
        //     double weight = i == 0 ? wc_0 : wc_i;
        //     std::cout << weight << std::endl;
        //     std::cout << mean_centered.col(i) * mean_centered.col(i).transpose() << std::endl;
        //     new_covariance += weight * mean_centered.col(i) * mean_centered.col(i).transpose();
        // }
        return StateAndCovariance();
    }

private: // members ////////////////////////////////////////////////////////////

    //
    // current state
    //
    State state;

    //
    // current covariance of the states
    //
    Covariance covariance;

    //
    // some precomputed parameters for the UT
    //
    ukf_params_t params;

    //
    // used to compute dt for updates, this may not work if some sensors take long
    // to process
    //
    Timestamp last_time;

};

}  // namespace
