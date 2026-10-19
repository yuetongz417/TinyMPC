// Fixed-QP Cartpole convergence experiment:
//   1. ordinary ADMM
//   2. Halpern ADMM (no restart)
//   3. Halpern ADMM with fixed-frequency restart
//   4. Halpern ADMM with residual-triggered adaptive restart
//
// All methods solve the SAME fixed QP from independently initialized states.
// They run the same number of raw ADMM operator evaluations (no early stop).
// Residuals are measured BEFORE Halpern averaging and use the same T.
//
// Prerequisite: src/tinympc/halpern_restart.hpp and .cpp exposing
// tiny_solve_halpern_fixed_restart_trace(...) from the restart package.

#include <tinympc/tiny_api.hpp>
#include <tinympc/admm.hpp>
#include <tinympc/halpern_restart.hpp>
#include <tinympc/halpern_adaptive.hpp>

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
constexpr int DEFAULT_RESTART_K = 2;
constexpr int DEFAULT_INITIAL_EPOCH_STEPS = 2;

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

tinytype fixed_point_residual(const TinyWorkspace* w,
                              const ADMMState& before) {
    return std::sqrt(
        (w->vnew - before.v).squaredNorm() +
        (w->znew - before.z).squaredNorm() +
        (w->g    - before.g).squaredNorm() +
        (w->y    - before.y).squaredNorm());
}

// Same Cartpole setup as your original fixed-problem diagnostic.
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

    tinyMatrix Adyn =
        Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn =
        Map<Matrix<tinytype, NSTATES, NINPUTS>>(Bdyn_data);
    StateVec fdyn = StateVec::Zero();
    StateVec Q = Map<StateVec>(Q_data);
    Matrix<tinytype, NINPUTS, 1> R =
        Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    const int status = tiny_setup(&solver, Adyn, Bdyn, fdyn,
                                  Q.asDiagonal(), R.asDiagonal(),
                                  rho_value, NSTATES, NINPUTS,
                                  NHORIZON, 0);
    if (status != 0 || solver == nullptr) {
        std::cerr << "tiny_setup failed, status = " << status << '\n';
        return nullptr;
    }

    // Effectively inactive state bounds; meaningful input box constraints.
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
    tiny_set_x0(solver, x0);  // ONLY ONCE: no closed-loop simulation.

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
    std::vector<int> restart_evals;
    int bad_adaptive_restarts = 0;
};

struct TraceContext {
    std::ofstream* csv;
    std::string method;
    RunSummary summary;
    tinytype epoch_start_residual = 0;
};

// Same CSV measurement for ALL methods. The restart solver copies its
// pre-evaluation slacks into w->v and w->z; the baselines do so too.
void record_evaluation(TinySolver* solver, TraceContext& ctx,
                       int eval, int epoch, int epoch_step,
                       bool restart_now, tinytype residual) {
    const TinyWorkspace* w = solver->work;
    const tinytype primal_inf = std::max(
        (w->x - w->vnew).cwiseAbs().maxCoeff(),
        (w->u - w->znew).cwiseAbs().maxCoeff());
    const tinytype slack_change_inf = solver->cache->rho * std::max(
        (w->vnew - w->v).cwiseAbs().maxCoeff(),
        (w->znew - w->z).cwiseAbs().maxCoeff());
    const tinytype input_violation = std::max(
        tinytype(0), w->u.cwiseAbs().maxCoeff() - INPUT_LIMIT);
    const tinytype max_abs_projected_input =
        w->znew.cwiseAbs().maxCoeff();

    RunSummary& s = ctx.summary;
    // At epoch_step=0, this is ||T(anchor)-anchor|| exactly.
    if (epoch_step == 0) ctx.epoch_start_residual = residual;
    const tinytype ratio = (ctx.epoch_start_residual > 0)
        ? residual / ctx.epoch_start_residual : tinytype(0);
    s.residuals.push_back(residual);
    if (eval == 1) s.first_residual = residual;
    s.final_residual = residual;
    if (s.first_below_tol < 0 && residual <= FP_TOL)
        s.first_below_tol = eval;
    if (input_violation > 1e-10)
        ++s.active_input_steps;
    if (restart_now) {
        ++s.restart_count;
        s.restart_evals.push_back(eval);
        if (ctx.method == "halpern_adaptive" && epoch > 0 &&
            ratio > std::exp(tinytype(-1)) + tinytype(1e-10))
            ++s.bad_adaptive_restarts;
        if (s.first_restart_eval < 0)
            s.first_restart_eval = eval;
    }

    (*ctx.csv) << ctx.method << ',' << eval << ',' << residual << ','
               << primal_inf << ',' << slack_change_inf << ','
               << input_violation << ',' << max_abs_projected_input << ','
               << epoch << ',' << epoch_step << ','
               << (restart_now ? 1 : 0) << ','
               << ctx.epoch_start_residual << ',' << ratio << '\n';

    if (eval == 1 || eval % 10 == 0 || eval == NUM_ITERATIONS || restart_now) {
        std::cout << ctx.method << " iter=" << eval
                  << " fp_residual=" << residual
                  << " primal_inf=" << primal_inf
                  << " input_violation=" << input_violation
                  << " epoch=" << epoch
                  << " epoch_step=" << epoch_step
                  << (restart_now ? " <RESTART>" : "") << '\n';
    }
}

// Original ADMM and original no-restart Halpern, with the same bookkeeping
// as the new solver (w->v and w->z are previous-slack scratch variables).
void trace_iterations(TinySolver* solver, bool halpern, TraceContext& ctx) {
    TinyWorkspace* w = solver->work;
    const ADMMState anchor = read_admm_state(w);

    for (int k = 0; k < NUM_ITERATIONS; ++k) {
        const ADMMState before = read_admm_state(w);
        w->v = before.v;
        w->z = before.z;
        apply_admm_operator(solver);
        const tinytype residual = fixed_point_residual(w, before);

        record_evaluation(solver, ctx, k + 1, 0, k, false, residual);

        // Do not average the final output. No early stopping in this test.
        if (halpern && k + 1 < NUM_ITERATIONS) {
            const tinytype beta = tinytype(1) / tinytype(k + 2);
            const tinytype alpha = tinytype(1) - beta;
            w->vnew = alpha * w->vnew + beta * anchor.v;
            w->znew = alpha * w->znew + beta * anchor.z;
            w->g    = alpha * w->g    + beta * anchor.g;
            w->y    = alpha * w->y    + beta * anchor.y;
        }
    }
}

// Called by halpern_restart.cpp after EACH raw T evaluation and BEFORE
// averaging/restarting. We only measure here; the solver owns the updates.
void restart_trace_callback(TinySolver* solver, int total_eval,
                            int epoch, int epoch_step, int restart_now,
                            tinytype residual, void* user_data) {
    auto* ctx = static_cast<TraceContext*>(user_data);
    record_evaluation(solver, *ctx, total_eval, epoch, epoch_step,
                      restart_now != 0, residual);
}

void print_summary(const TraceContext& ctx) {
    const RunSummary& s = ctx.summary;
    std::cout << ctx.method
              << ": first=" << s.first_residual
              << " first_below_tol=" << s.first_below_tol
              << " final=" << s.final_residual
              << " input_violation_steps=" << s.active_input_steps
              << " restarts=" << s.restart_count
              << " first_restart_eval=" << s.first_restart_eval
              << " bad_adaptive_restarts=" << s.bad_adaptive_restarts
              << '\n';
    if (!s.restart_evals.empty()) {
        std::cout << "  restart evaluations:";
        for (int k : s.restart_evals) std::cout << ' ' << k;
        std::cout << '\n';
    }
}

int main(int argc, char** argv) {
    // Optional argument: K, the number of Halpern updates per epoch.
    int K = DEFAULT_RESTART_K;
    int first_epoch_steps = DEFAULT_INITIAL_EPOCH_STEPS;
    if (argc > 1) {
        char* end = nullptr;
        const long value = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || value < 1 || value > 100000) {
            std::cerr << "Usage: cartpole_example_modified [positive_K] [positive_initial_epoch_steps]\n";
            return 1;
        }
        K = static_cast<int>(value);
    }
    if (argc > 2) {
        char* end = nullptr;
        const long value = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || value < 1 || value > 100000) {
            std::cerr << "Usage: cartpole_example_modified [positive_K] "
                         "[positive_initial_epoch_steps]\n";
            return 1;
        }
        first_epoch_steps = static_cast<int>(value);
    }
    if (argc > 3) {
        std::cerr << "Usage: cartpole_example_modified [positive_K] "
                     "[positive_initial_epoch_steps]\n";
        return 1;
    }

    const std::string csv_path =
        "result/cartpole_adaptive_K" + std::to_string(K) +
        "_init" + std::to_string(first_epoch_steps) + ".csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path
                  << ". Run 'mkdir -p result' from the repository root.\n";
        return 1;
    }
    csv << std::setprecision(16);
    std::cout << std::setprecision(10);
    csv << "method,iteration,fp_residual,primal_inf,rho_slack_change_inf,"
           "input_violation,max_abs_projected_input,epoch,epoch_step,restart,"
           "epoch_start_residual,epoch_residual_ratio\n";

    TinySolver* admm_solver = make_cartpole_solver();
    TinySolver* halpern_solver = make_cartpole_solver();
    TinySolver* restart_solver = make_cartpole_solver();
    TinySolver* adaptive_solver = make_cartpole_solver();
    if (!admm_solver || !halpern_solver || !restart_solver || !adaptive_solver) {
        std::cerr << "Failed to initialize all four solvers.\n";
        return 1;
    }

    TraceContext admm{&csv, "admm", {}};
    TraceContext halpern{&csv, "halpern", {}};
    TraceContext restarted{&csv, "halpern_restart", {}};
    TraceContext adaptive{&csv, "halpern_adaptive", {}};

    trace_iterations(admm_solver, false, admm);
    trace_iterations(halpern_solver, true, halpern);

    // stop_on_convergence = 0 so all methods use exactly NUM_ITERATIONS
    // raw T evaluations, INCLUDING the extra operator steps at restarts.
    const int status = tiny_solve_halpern_fixed_restart_trace(
        restart_solver, K, 0, restart_trace_callback, &restarted);
    if (status != 1) {
        std::cerr << "Unexpected restarted-solver return code: " << status
                  << " (expected 1 because early stopping is disabled).\n";
        return 1;
    }

    // Method 4: Eq. (10) adaptive restart, initial epoch of
    // first_epoch_steps averages and later epochs triggered by 1/e reduction.
    const int adaptive_status = tiny_solve_halpern_adaptive_restart_trace(
        adaptive_solver, first_epoch_steps, 0,
        restart_trace_callback, &adaptive);
    if (adaptive_status != 1) {
        std::cerr << "Unexpected adaptive return code: " << adaptive_status
                  << " (expected 1 with early stopping disabled).\n";
        return 1;
    }

    std::cout << "\n--- Fixed Cartpole QP, tolerance = " << FP_TOL
              << ", fixed K = " << K
              << ", adaptive initial epoch = " << first_epoch_steps
              << " ---\n";
    print_summary(admm);
    print_summary(halpern);
    print_summary(restarted);
    print_summary(adaptive);

    // Correctness checks: same T(w0), and identical Halpern path through
    // the first restart-triggering raw T step (eval K+1).
    tinytype first_difference = std::max(
        std::abs(admm.summary.first_residual - halpern.summary.first_residual),
        std::max(
            std::abs(admm.summary.first_residual - restarted.summary.first_residual),
            std::abs(admm.summary.first_residual - adaptive.summary.first_residual)));
    if (admm.summary.residuals.size() != NUM_ITERATIONS ||
        halpern.summary.residuals.size() != NUM_ITERATIONS ||
        restarted.summary.residuals.size() != NUM_ITERATIONS ||
        adaptive.summary.residuals.size() != NUM_ITERATIONS) {
        std::cerr << "ERROR: missing evaluations in at least one trace.\n";
        return 2;
    }
    tinytype initial_halpern_difference = 0;
    const int equal_evals = std::min(NUM_ITERATIONS, K + 1);
    for (int i = 0; i < equal_evals; ++i) {
        initial_halpern_difference = std::max(
            initial_halpern_difference,
            std::abs(halpern.summary.residuals[i] -
                     restarted.summary.residuals[i]));
    }
    std::cout << "First-evaluation residual disagreement: "
              << first_difference << '\n';
    std::cout << "Max Halpern/restart disagreement through eval "
              << equal_evals << ": " << initial_halpern_difference << '\n';
    if (K + 1 < NUM_ITERATIONS) {
        std::cout << "First restart: expected eval=" << K + 1
                  << ", observed=" << restarted.summary.first_restart_eval
                  << '\n';
    }
    // In the first adaptive epoch, there are first_epoch_steps Halpern
    // updates and then a raw T restart step: identical to plain Halpern
    // through evaluation first_epoch_steps+1.
    tinytype adaptive_first_epoch_difference = 0;
    const int adaptive_equal = std::min(NUM_ITERATIONS, first_epoch_steps + 1);
    for (int i = 0; i < adaptive_equal; ++i) {
        adaptive_first_epoch_difference = std::max(
            adaptive_first_epoch_difference,
            std::abs(halpern.summary.residuals[i] - adaptive.summary.residuals[i]));
    }
    std::cout << "Max Halpern/adaptive disagreement through eval "
              << adaptive_equal << ": " << adaptive_first_epoch_difference << '\n';
    if (first_epoch_steps + 1 < NUM_ITERATIONS &&
        adaptive.summary.first_restart_eval != first_epoch_steps + 1) {
        std::cerr << "ERROR: adaptive first restart did not match first epoch length.\n";
        return 2;
    }
    if (adaptive.summary.bad_adaptive_restarts != 0) {
        std::cerr << "ERROR: adaptive restart violates 1/e threshold.\n";
        return 2;
    }
    std::cout << "Saved " << csv_path << '\n';

    if (first_difference > 1e-10 || initial_halpern_difference > 1e-10 ||
        adaptive_first_epoch_difference > 1e-10) {
        std::cerr << "ERROR: initial trajectories differ; check solver state.\n";
        return 2;
    }
    return 0;
}
