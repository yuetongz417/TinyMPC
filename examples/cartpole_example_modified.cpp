// Fixed-problem convergence diagnostic for TinyMPC ADMM and Halpern ADMM.
// Derived from examples/cartpole_example.cpp. This intentionally runs the
// iteration kernels directly (without early stopping) to log ||T(w)-w||.

#include <tinympc/tiny_api.hpp>
#include <tinympc/admm.hpp>
#include <tinympc/halpern_restart.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define NSTATES 4
#define NINPUTS 1
#define NHORIZON 10

using StateVec = Matrix<tinytype, NSTATES, 1>;
using StateTraj = Matrix<tinytype, NSTATES, NHORIZON>;
using InputTraj = Matrix<tinytype, NINPUTS, NHORIZON - 1>;

constexpr int NUM_ITERATIONS = 200;
constexpr tinytype INPUT_LIMIT = 0.5;
constexpr tinytype FP_TOL = 1e-4;
constexpr int DEFAULT_RESTART_K = 20;

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


tinytype fixed_point_residual(
    const TinyWorkspace* w,
    const ADMMState& before
) {
    return std::sqrt(
        (w->vnew - before.v).squaredNorm() +
        (w->znew - before.z).squaredNorm() +
        (w->g - before.g).squaredNorm() +
        (w->y - before.y).squaredNorm()
    );
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
    tinytype first_residual = 0;
    tinytype final_residual = 0;

    int first_below_tol = -1;
    int active_input_steps = 0;

    int restart_count = 0;
    int first_restart_eval = -1;

    std::vector<tinytype> residuals;
};

struct TraceContext {
    std::ofstream* csv;
    std::string method;
    RunSummary summary;
};

void record_evaluation(
    TinySolver* solver,
    TraceContext& ctx,
    int eval,
    int epoch,
    int epoch_step,
    bool restart_now,
    tinytype residual
) {
    const TinyWorkspace* w = solver->work;

    // Primal equality residual.
    const tinytype primal_inf = std::max(
        (w->x - w->vnew).cwiseAbs().maxCoeff(),
        (w->u - w->znew).cwiseAbs().maxCoeff()
    );

    // Scaled change in slack variables.
    const tinytype slack_change_inf =
        solver->cache->rho * std::max(
            (w->vnew - w->v).cwiseAbs().maxCoeff(),
            (w->znew - w->z).cwiseAbs().maxCoeff()
        );

    // Violation of raw control inputs.
    const tinytype input_violation = std::max(
        tinytype(0),
        w->u.cwiseAbs().maxCoeff() - INPUT_LIMIT
    );

    // Largest magnitude of projected input.
    const tinytype max_abs_projected_input =
        w->znew.cwiseAbs().maxCoeff();

    // Update summary.
    RunSummary& s = ctx.summary;

    s.residuals.push_back(residual);

    if (eval == 1)
        s.first_residual = residual;

    s.final_residual = residual;

    if (s.first_below_tol < 0 && residual <= FP_TOL)
        s.first_below_tol = eval;

    if (input_violation > 1e-10)
        ++s.active_input_steps;

    if (restart_now) {
        ++s.restart_count;

        if (s.first_restart_eval < 0)
            s.first_restart_eval = eval;
    }

    // Write CSV row.
    (*ctx.csv)
        << ctx.method << ','
        << eval << ','
        << residual << ','
        << primal_inf << ','
        << slack_change_inf << ','
        << input_violation << ','
        << max_abs_projected_input << ','
        << epoch << ','
        << epoch_step << ','
        << (restart_now ? 1 : 0)
        << '\n';

    // Print selected iterations.
    if (
        eval == 1 ||
        eval % 10 == 0 ||
        eval == NUM_ITERATIONS ||
        restart_now
    ) {
        std::cout
            << ctx.method
            << " iter=" << eval
            << " fp_residual=" << residual
            << " primal_inf=" << primal_inf
            << " input_violation=" << input_violation
            << " epoch=" << epoch
            << " epoch_step=" << epoch_step;

        if (restart_now)
            std::cout << " <RESTART>";

        std::cout << '\n';
    }
}

void trace_iterations(
    TinySolver* solver,
    bool halpern,
    TraceContext& ctx
) {
    TinyWorkspace* w = solver->work;

    // Original anchor w^0.
    const ADMMState anchor = read_admm_state(w);

    for (int k = 0; k < NUM_ITERATIONS; ++k) {

        // Save state before applying T.
        const ADMMState before = read_admm_state(w);

        // Keep previous slack arrays consistent.
        w->v = before.v;
        w->z = before.z;

        // Raw ADMM evaluation.
        apply_admm_operator(solver);

        // Compute ||T(w^k) - w^k||.
        const tinytype residual =
            fixed_point_residual(w, before);

        // Record raw output before averaging.
        record_evaluation(
            solver,
            ctx,
            k + 1,
            0,
            k,
            false,
            residual
        );

        // Halpern averaging.
        // Ordinary ADMM skips this block.
        //
        // Do not average after the final evaluation.

        if (halpern && k + 1 < NUM_ITERATIONS) {
            const tinytype beta =
                tinytype(1) / tinytype(k + 2);

            const tinytype alpha =
                tinytype(1) - beta;

            w->vnew =
                alpha * w->vnew + beta * anchor.v;

            w->znew =
                alpha * w->znew + beta * anchor.z;

            w->g =
                alpha * w->g + beta * anchor.g;

            w->y =
                alpha * w->y + beta * anchor.y;
        }
    }
}

void restart_trace_callback(
    TinySolver* solver,
    int total_eval,
    int epoch,
    int epoch_step,
    int restart_now,
    tinytype residual,
    void* user_data
) {
    auto* ctx =
        static_cast<TraceContext*>(user_data);

    record_evaluation(
        solver,
        *ctx,
        total_eval,
        epoch,
        epoch_step,
        restart_now != 0,
        residual
    );
}


// ============================================================
// 7. PRINT SUMMARY
// ============================================================

void print_summary(const TraceContext& ctx) {
    const RunSummary& s = ctx.summary;

    std::cout
        << ctx.method
        << ": first=" << s.first_residual
        << " first_below_tol=" << s.first_below_tol
        << " final=" << s.final_residual
        << " input_violation_steps="
        << s.active_input_steps
        << " restarts=" << s.restart_count
        << " first_restart_eval="
        << s.first_restart_eval
        << '\n';
}

int main(int argc, char** argv) {

    // Restart interval K.
    // Default is 20 Halpern updates per epoch.
    int K = DEFAULT_RESTART_K;

    if (argc > 1) {
        char* end = nullptr;

        const long value =
            std::strtol(argv[1], &end, 10);

        if (
            end == argv[1] ||
            *end != '\0' ||
            value < 1 ||
            value > 100000
        ) {
            std::cerr
                << "Usage: cartpole_example_modified "
                   "[positive_K]\n";
            return 1;
        }

        K = static_cast<int>(value);
    }

    // Save results separately for each K.
    const std::string csv_path =
        "result/cartpole_restart_K" +
        std::to_string(K) +
        ".csv";

    std::ofstream csv(csv_path);

    if (!csv) {
        std::cerr
            << "Cannot open " << csv_path
            << ". Run 'mkdir -p result' "
               "from the repository root.\n";
        return 1;
    }

    csv << std::setprecision(16);
    std::cout << std::setprecision(10);

    csv
        << "method,iteration,fp_residual,primal_inf,"
           "rho_slack_change_inf,input_violation,"
           "max_abs_projected_input,epoch,epoch_step,"
           "restart\n";

    // Create three independent, identical solvers.
    TinySolver* admm_solver =
        make_cartpole_solver();

    TinySolver* halpern_solver =
        make_cartpole_solver();

    TinySolver* restart_solver =
        make_cartpole_solver();

    if (
        !admm_solver ||
        !halpern_solver ||
        !restart_solver
    ) {
        std::cerr
            << "Failed to initialize all three solvers.\n";
        return 1;
    }

    // Create trace contexts.
    TraceContext admm{
        &csv,
        "admm",
        {}
    };

    TraceContext halpern{
        &csv,
        "halpern",
        {}
    };

    TraceContext restarted{
        &csv,
        "halpern_restart",
        {}
    };

    // --------------------------------------------------------
    // Method 1: Ordinary ADMM
    // --------------------------------------------------------

    trace_iterations(
        admm_solver,
        false,
        admm
    );

    // --------------------------------------------------------
    // Method 2: Halpern ADMM without restart
    // --------------------------------------------------------

    trace_iterations(
        halpern_solver,
        true,
        halpern
    );

    // --------------------------------------------------------
    // Method 3: Fixed-frequency restarted Halpern ADMM
    // --------------------------------------------------------

    // stop_on_convergence = 0:
    // Run the full 200-evaluation budget.
    //
    // Restart steps are INCLUDED in the evaluation count.

    const int status =
        tiny_solve_halpern_fixed_restart_trace(
            restart_solver,
            K,
            0,
            restart_trace_callback,
            &restarted
        );

    // With early stopping disabled, the restart solver
    // should finish the entire budget and return 1.
    if (status != 1) {
        std::cerr
            << "Unexpected restarted solver return code: "
            << status
            << " (expected 1 because early stopping "
               "is disabled).\n";

        return 1;
    }

    // ========================================================
    // 9. PRINT EXPERIMENT RESULTS
    // ========================================================

    std::cout
        << "\n--- Fixed Cartpole QP ---\n"
        << "Tolerance = " << FP_TOL << '\n'
        << "Restart K = " << K << '\n'
        << "Evaluation budget = "
        << NUM_ITERATIONS << "\n\n";

    print_summary(admm);
    print_summary(halpern);
    print_summary(restarted);

    // ========================================================
    // 10. CORRECTNESS CHECKS
    // ========================================================

    // The first residual must be identical across all methods.
    const tinytype first_difference = std::max(
        std::abs(
            admm.summary.first_residual -
            halpern.summary.first_residual
        ),
        std::abs(
            admm.summary.first_residual -
            restarted.summary.first_residual
        )
    );

    // Verify callbacks ran for the full budget before indexing traces.
    if (admm.summary.residuals.size() != NUM_ITERATIONS ||
        halpern.summary.residuals.size() != NUM_ITERATIONS ||
        restarted.summary.residuals.size() != NUM_ITERATIONS) {
        std::cerr << "ERROR: one method did not record all "
                  << NUM_ITERATIONS << " operator evaluations.\n";
        return 2;
    }

    // Halpern and restarted Halpern must agree
    // through the first restart evaluation (K+1).
    tinytype initial_halpern_difference = 0;

    const int equal_evals =
        std::min(NUM_ITERATIONS, K + 1);

    for (int i = 0; i < equal_evals; ++i) {
        initial_halpern_difference = std::max(
            initial_halpern_difference,
            std::abs(
                halpern.summary.residuals[i] -
                restarted.summary.residuals[i]
            )
        );
    }

    std::cout
        << "First-evaluation residual disagreement: "
        << first_difference << '\n';

    std::cout
        << "Max Halpern/restart disagreement through eval "
        << equal_evals << ": "
        << initial_halpern_difference << '\n';

    if (K + 1 < NUM_ITERATIONS) {
        std::cout
            << "First restart: expected eval="
            << K + 1
            << ", observed="
            << restarted.summary.first_restart_eval
            << '\n';
        if (restarted.summary.first_restart_eval != K + 1) {
            std::cerr << "ERROR: first restart happened at the wrong evaluation.\n";
            return 2;
        }
    }

    std::cout
        << "Saved " << csv_path << '\n';

    if (
        first_difference > 1e-10 ||
        initial_halpern_difference > 1e-10
    ) {
        std::cerr
            << "ERROR: initial trajectories differ; "
               "check solver state.\n";
        return 2;
    }

    return 0;
}