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
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        orientation = Eigen::Matrix3d::Identity();
        angular_vel = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        gyro_bias = Eigen::Vector3d::Zero();
    }

    // makes the sigma point transformation pretty easy
    State operator+(const Eigen::Matrix<double, states::NUM_STATES, 1>& lhs) const
    {
        State new_state;
        Eigen::Vector3d lie_rotation(ln(orientation));      
        new_state.position = position + lhs.block<3, 1>(states::X, 0);
        new_state.velocity = velocity + lhs.block<3, 1>(states::VX, 0);
        new_state.orientation = exp(lie_rotation + lhs.block<3, 1>(states::RX, 0));
        new_state.angular_vel = angular_vel + lhs.block<3, 1>(states::WX, 0);
        new_state.acc_bias = acc_bias + lhs.block<3, 1>(states::AX_b, 0);
        new_state.gyro_bias = gyro_bias + lhs.block<3, 1>(states::GX_b, 0);

        return new_state;
    };

    void print() const
    {
        std::cout << "position:           " << position.transpose() << std::endl;
        std::cout << "velocity:           " << velocity.transpose() << std::endl;
        std::cout << "orientation (x vec) " << orientation.block<3, 1>(0, 0).transpose()<< std::endl;
        std::cout << "angular_vel:        " << angular_vel.transpose()<< std::endl;
        std::cout << "acc_bias:           " << acc_bias.transpose() << std::endl;
        std::cout << "gyro_bias:          " << gyro_bias.transpose() << std::endl;
    }
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
    ukf_params_t(double alpha_=1E-2, double beta_=2, double kappa_=0)
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
    
public:

    void test_UT(const State &initial_state, const Covariance &initial_covariance)
    {
        // Let's do the transform now (not really covariance type, but close enough)
        Covariance cov_sqrt = initial_covariance.llt().matrixL();
        cov_sqrt *= params.sqrt_cov_factor;

        // populate sigma vector
        SigmaPoints sigmas(NUM_SIGMA_POINTS);
        sigmas[0] = initial_state;
        for (size_t i = 0; i < NUM_STATES; ++i)
        {
            // since sigma[0] is taken, add one to the index
            sigmas[2 * i + 1] = initial_state + cov_sqrt.row(i);
            sigmas[2 * i + 2] = initial_state + -cov_sqrt.row(i);
        }

        // insert my fancy transformation function here //
        
        //
        // Strategy:
        // Compute average rotation then compute mean/cov of the entire set of sigma points
        // (using the rotation's Lie Algebra)
        //
        std::vector<Eigen::Matrix3d> rotations;
        rotations.reserve(NUM_SIGMA_POINTS);
        for(const State& sigma : sigmas)
        {
            rotations.push_back(sigma.orientation); 
        }
        
        // Compute mean
        double mean_weight = params.mean_weight.first;
        State new_state;
        for (const State& sigma_point : sigmas)
        {
            new_state.position += mean_weight * sigma_point.position;
            new_state.velocity += mean_weight * sigma_point.velocity;
            new_state.angular_vel += mean_weight * sigma_point.angular_vel;
            new_state.acc_bias += mean_weight * sigma_point.acc_bias;
            new_state.gyro_bias += mean_weight * sigma_point.gyro_bias;

            mean_weight = params.mean_weight.second;
        }
        Eigen::Vector3d lie_mean = average_rotations(rotations);
        new_state.orientation = exp(lie_mean);

        // Compute covariance
        double cov_weight = params.cov_weight.first;
        Covariance new_covariance = Covariance::Zero();
        for (const State& sigma_point : sigmas)
        {
            Eigen::Matrix<double, NUM_STATES, 1> err;
            err.block<3, 1>(states::X, 0) = sigma_point.position - new_state.position;
            err.block<3, 1>(states::VX, 0) = sigma_point.velocity - new_state.velocity;
            err.block<3, 1>(states::RX, 0) = ln(sigma_point.orientation) - lie_mean;
            err.block<3, 1>(states::WX, 0) = sigma_point.angular_vel - new_state.angular_vel;
            err.block<3, 1>(states::AX_b, 0) = sigma_point.acc_bias - new_state.acc_bias;
            err.block<3, 1>(states::GX_b, 0) = sigma_point.gyro_bias - new_state.gyro_bias;

            new_covariance += cov_weight * err * err.transpose();
            cov_weight = params.cov_weight.second;
        }

        std::cout << "State" << std::endl;
        new_state.print();
        std::cout << "Covariance" << std::endl << new_covariance << std::endl;
    }

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
        // Let's do the transform now (not really covariance type, but close enough)
        Covariance cov_sqrt = initial_covariance.llt().matrixL();
        cov_sqrt *= params.sqrt_cov_factor;

        // populate sigma vector
        SigmaPoints sigmas(NUM_SIGMA_POINTS);
        sigmas[0] = initial_state;
        for (size_t i = 0; i < NUM_STATES; ++i)
        {
            // since sigma[0] is taken, add one to the index
            sigmas[2 * i + 1] = initial_state + cov_sqrt.row(i);
            sigmas[2 * i + 2] = initial_state + -cov_sqrt.row(i);
        }

        // this function takes a SigmaPoints vector and transforms each point
        transition_f(sigmas);
        
        //
        // TODO: Redo the rotation stuff using more operator overloading to make it cleaner
        //
        std::vector<Eigen::Matrix3d> rotations;
        rotations.reserve(NUM_SIGMA_POINTS);
        for(const State& sigma : sigmas)
        {
            rotations.push_back(sigma.orientation); 
        }
        
        // Compute mean
        double mean_weight = params.mean_weight.first;
        State new_state;
        for (const State& sigma_point : sigmas)
        {
            new_state.position += mean_weight * sigma_point.position;
            new_state.velocity += mean_weight * sigma_point.velocity;
            new_state.angular_vel += mean_weight * sigma_point.angular_vel;
            new_state.acc_bias += mean_weight * sigma_point.acc_bias;
            new_state.gyro_bias += mean_weight * sigma_point.gyro_bias;

            mean_weight = params.mean_weight.second;
        }
        Eigen::Vector3d lie_mean = average_rotations(rotations);
        new_state.orientation = exp(lie_mean);

        // Compute covariance
        double cov_weight = params.cov_weight.first;
        Covariance new_covariance = Covariance::Zero();
        for (const State& sigma_point : sigmas)
        {
            Eigen::Matrix<double, NUM_STATES, 1> err;
            err.block<3, 1>(states::X, 0) = sigma_point.position - new_state.position;
            err.block<3, 1>(states::VX, 0) = sigma_point.velocity - new_state.velocity;
            err.block<3, 1>(states::RX, 0) = ln(sigma_point.orientation) - lie_mean;
            err.block<3, 1>(states::WX, 0) = sigma_point.angular_vel - new_state.angular_vel;
            err.block<3, 1>(states::AX_b, 0) = sigma_point.acc_bias - new_state.acc_bias;
            err.block<3, 1>(states::GX_b, 0) = sigma_point.gyro_bias - new_state.gyro_bias;

            new_covariance += cov_weight * err * err.transpose();
            cov_weight = params.cov_weight.second;
        }

        StateAndCovariance s_c;
        s_c.state = new_state;
        s_c.covariance = new_covariance;

        return s_c;
    }

    //
    // compute the average of a set of rotation matricies
    // 
    Eigen::Vector3d average_rotations(std::vector<Eigen::Matrix3d> rots) const
    {
        Eigen::Matrix3d u = rots[0];
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

        for (size_t iters = 0; iters < 5; ++iters)
        {
            Eigen::MatrixXd v(3, rots.size());
            Eigen::Matrix3d u_inv = u.inverse();
            cov = Eigen::Matrix3d::Zero();

            // build the v matrix
            for (size_t i = 0; i < rots.size(); ++i)
            {
                v.col(i) = ln(rots[i] * u_inv);
            }

            // compute covariance and mean
            Eigen::Vector3d tangent_space_avg = Eigen::Vector3d::Zero();
            const double average_factor = 1.0 / rots.size();
            for (size_t i = 0; i < rots.size(); ++i)
            {
                double cov_weight = params.cov_weight.second;
                if (i == 0)
                    cov_weight = params.cov_weight.first;

                double mean_weight = params.mean_weight.second;
                if (i == 0)
                    cov_weight = params.mean_weight.first;

                cov += average_factor * cov_weight * v.col(i) * v.col(i).transpose();
                tangent_space_avg += mean_weight * v.col(i);
            }
            u = exp(tangent_space_avg) * u; 
        }
        return ln(u);
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
