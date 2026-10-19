// Fixed-problem convergence diagnostic for TinyMPC ADMM and Halpern ADMM.
// Derived from examples/cartpole_example.cpp. This intentionally runs the
// iteration kernels directly (without early stopping) to log ||T(w)-w||.

#include <tinympc/tiny_api.hpp>
#include <tinympc/admm.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define NSTATES 4
#define NINPUTS 1
#define NHORIZON 10

using StateVec = Matrix<tinytype, NSTATES, 1>;
using StateTraj = Matrix<tinytype, NSTATES, NHORIZON>;
using InputTraj = Matrix<tinytype, NINPUTS, NHORIZON - 1>;

constexpr int NUM_ITERATIONS = 200;
constexpr tinytype INPUT_LIMIT = 0.5;
constexpr tinytype FP_TOL = 1e-4;

struct ADMMState {
    tinyMatrix v, z, g, y;
};

ADMMState read_admm_state(const TinyWorkspace* w) {
    return {w->vnew, w->znew, w->g, w->y};
}

void apply_admm_operator(TinySolver* solver) {
    update_linear_cost(solver);
    backward_pass_grad(solver);
    forward_pass(solver);
    update_slack(solver);
    update_dual(solver);
}

// Both solvers have identical data and zero-initialized ADMM states.
TinySolver* make_cartpole_solver() {
    TinySolver* solver = nullptr;
    const tinytype rho_value = 1.0;

    tinytype Adyn_data[NSTATES * NSTATES] = {
        1.0, 0.01, 0.0, 0.0,
        0.0, 1.0, 0.039, 0.0,
        0.0, 0.0, 1.002, 0.01,
        0.0, 0.0, 0.458, 1.002
    };
    tinytype Bdyn_data[NSTATES * NINPUTS] = {0.0, 0.02, 0.0, 0.067};
    tinytype Q_data[NSTATES] = {10.0, 1.0, 10.0, 1.0};
    tinytype R_data[NINPUTS] = {1.0};

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS>>(Bdyn_data);
    StateVec fdyn = StateVec::Zero();
    StateVec Q = Map<StateVec>(Q_data);
    Matrix<tinytype, NINPUTS, 1> R =
        Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    int status = tiny_setup(&solver, Adyn, Bdyn, fdyn,
                            Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 0);
    if (status != 0 || solver == nullptr) {
        std::cerr << "tiny_setup failed, status = " << status << "\n";
        return nullptr;
    }

    // State constraints are effectively inactive; input constraints are real.
    tinyMatrix x_min = StateTraj::Constant(-1e17);
    tinyMatrix x_max = StateTraj::Constant( 1e17);
    tinyMatrix u_min = InputTraj::Constant(-INPUT_LIMIT);
    tinyMatrix u_max = InputTraj::Constant( INPUT_LIMIT);
    if (tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max) != 0) {
        std::cerr << "tiny_set_bound_constraints failed\n";
        return nullptr;
    }

    solver->settings->adaptive_rho = 0;
    solver->settings->en_state_bound = 1;
    solver->settings->en_input_bound = 1;
    solver->settings->max_iter = NUM_ITERATIONS;

    StateVec x0;
    x0 << 0.5, 0.0, 0.0, 0.0;
    tiny_set_x0(solver, x0); // Only once: do NOT simulate the plant.

    StateVec Xref_origin;
    Xref_origin << 1.0, 0.0, 0.0, 0.0;
    solver->work->Xref = Xref_origin.replicate(1, NHORIZON);
    return solver;
}

struct RunSummary {
    tinytype first_residual;
    tinytype final_residual;
    int first_below_tol;
    int active_input_steps;
};

RunSummary trace_iterations(TinySolver* solver, bool halpern,
                            std::ofstream& csv) {
    TinyWorkspace* w = solver->work;
    const ADMMState anchor = read_admm_state(w); // fixed w^0
    const std::string method = halpern ? "halpern" : "admm";

    RunSummary summary{0, 0, -1, 0};

    for (int k = 0; k < NUM_ITERATIONS; ++k) {
        const ADMMState before = read_admm_state(w);

        // Raw evaluation of T(w^k), exactly as in admm.cpp / halpern.cpp.
        apply_admm_operator(solver);

        const tinytype residual = std::sqrt(
            (w->vnew - before.v).squaredNorm() +
            (w->znew - before.z).squaredNorm() +
            (w->g    - before.g).squaredNorm() +
            (w->y    - before.y).squaredNorm());

        // Diagnostics on RAW output (before Halpern averaging).
        const tinytype primal_inf = std::max(
            (w->x - w->vnew).cwiseAbs().maxCoeff(),
            (w->u - w->znew).cwiseAbs().maxCoeff());
        const tinytype slack_change_inf = solver->cache->rho * std::max(
            (w->vnew - before.v).cwiseAbs().maxCoeff(),
            (w->znew - before.z).cwiseAbs().maxCoeff());
        const tinytype input_violation = std::max(
            tinytype(0), w->u.cwiseAbs().maxCoeff() - INPUT_LIMIT);
        const tinytype max_abs_projected_input = w->znew.cwiseAbs().maxCoeff();

        if (k == 0) summary.first_residual = residual;
        summary.final_residual = residual;
        if (summary.first_below_tol < 0 && residual <= FP_TOL)
            summary.first_below_tol = k + 1;
        if (input_violation > 1e-10) ++summary.active_input_steps;

        csv << method << ',' << k + 1 << ',' << residual << ','
            << primal_inf << ',' << slack_change_inf << ','
            << input_violation << ',' << max_abs_projected_input << '\n';

        if (k == 0 || (k + 1) % 10 == 0 || k + 1 == NUM_ITERATIONS) {
            std::cout << method << " iter=" << (k + 1)
                      << " fp_residual=" << residual
                      << " primal_inf=" << primal_inf
                      << " input_violation=" << input_violation << '\n';
        }

        // No termination here; log a full comparable residual trajectory.
        if (halpern && k + 1 < NUM_ITERATIONS) {
            const tinytype beta = tinytype(1) / tinytype(k + 2);
            const tinytype alpha = tinytype(1) - beta;
            w->vnew = alpha * w->vnew + beta * anchor.v;
            w->znew = alpha * w->znew + beta * anchor.z;
            w->g    = alpha * w->g    + beta * anchor.g;
            w->y    = alpha * w->y    + beta * anchor.y;
        }
    }
    return summary;
}

int main() {
    // From the TinyMPC repository root, first run: mkdir -p result
    std::ofstream csv("result/cartpole_convergence.csv");
    if (!csv) {
        std::cerr << "Cannot open result/cartpole_convergence.csv. "
                  << "Run 'mkdir -p result' from the repository root.\n";
        return 1;
    }
    csv << std::setprecision(16);
    std::cout << std::setprecision(10);
    csv << "method,iteration,fp_residual,primal_inf,rho_slack_change_inf,"
           "input_violation,max_abs_projected_input\n";

    TinySolver* admm_solver = make_cartpole_solver();
    TinySolver* halpern_solver = make_cartpole_solver();
    if (!admm_solver || !halpern_solver) return 1;

    const RunSummary admm = trace_iterations(admm_solver, false, csv);
    const RunSummary halpern = trace_iterations(halpern_solver, true, csv);

    std::cout << "\nFirst residual: ADMM=" << admm.first_residual
              << " Halpern=" << halpern.first_residual << '\n';
    std::cout << "First evaluation with residual <= " << FP_TOL
              << ": ADMM=" << admm.first_below_tol
              << " Halpern=" << halpern.first_below_tol
              << " (-1 means not reached)\n";
    std::cout << "Final residual: ADMM=" << admm.final_residual
              << " Halpern=" << halpern.final_residual << '\n';
    std::cout << "Evaluations with unprojected input outside the box: "
              << "ADMM=" << admm.active_input_steps
              << " Halpern=" << halpern.active_input_steps << '\n';
    std::cout << "Saved result/cartpole_convergence.csv\n";
    return 0;
}
