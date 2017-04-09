#include <iostream>
#include <Eigen/Dense>
#include <chrono>

namespace ukf
{

//
// Some basic typedefs not really related to the filter
//
using Clock     = std::chrono::high_resolution_clock;
using Timestamp = Clock::time_point;

enum states : uint8_t
{
    X,  // m
    Y,
    Z,
    VX,  // m/s
    VY,
    VZ,
    RX,  // rads, I'd like to get rid of Euler math
    RY,
    RZ,
    WX,  // rads/s
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

// Things preventing Quaternion solution:
//   - How to covariance with Quaternion?
//     * how to represent "rotational variance" in the matrix
//

// This is an alternate way to keep track of the state...
struct State
{
    Eigen::Vector3d    position;
    Eigen::Vector3d    velocity;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d    angular_vel;
    Eigen::Vector3d    acc_bias;
    Eigen::Vector3d    gyro_bias;
};

// Only time will tell if this is an easier way
//using State      = Eigen::Matrix<double, states::NUM_STATES, 1>;
using Covariance = Eigen::Matrix<double, states::NUM_STATES, states::NUM_STATES>;

struct StateAndCovariance
{
    State      state;
    Covariance covariance;
};

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
    };

    // normal parameters
    double alpha;
    double beta;
    double kappa;
    double lambda;    

    // weights for computing sigma points
    std::pair<double, double> mean_weight;
    std::pair<double, double> cov_weight;

};

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
    // Called with a new 
    //
    
private: // methods ////////////////////////////////////////////////////////////
    
    //
    //
    //
    template <typename Func>
    StateAndCovariance unscented_transform(const State &      initial_state, 
                                           const Covariance & initial_covariance,
                                           const Func         transition_f) const
    {
        
        // LDLT is better I think
        const Eigen::LLT<Covariance> cov_sqrt(initial_covariance.llt().matrixL());


        // There should be 2*n + 1 sigma points
        const size_t num_sigma_points = 2 * states::NUM_STATES + 1;

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
