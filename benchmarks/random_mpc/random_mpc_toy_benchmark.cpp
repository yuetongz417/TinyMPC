// Fixed QP comparison for one saved random_problems/prob_* instance.
// Build with -DMPC_BENCH_PROBLEM_DIR=/absolute/path/to/prob_Nh_4.
// Usage: random_mpc_fixed_benchmark [K=2] [max_evals=500] [output.csv]
// Raw ADMM operator evaluations are counted identically for both methods.

#include <tinympc/tiny_api.hpp>
#include <tinympc/admm.hpp>
#include <tinympc/halpern_reflected.hpp>

#ifndef PROGMEM
#define PROGMEM
#endif

#include "constants.hpp"
#include "rand_prob_tinympc_params.hpp"
#include "rand_prob_tinympc_xbar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using RowMajorMatrix = Eigen::Matrix<tinytype, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

struct Point {
    tinyMatrix v, z, g, y;
};

Point read_point(const TinyWorkspace* w) {
    return {w->vnew, w->znew, w->g, w->y};
}

void apply_raw_admm(TinySolver* solver) {
    update_linear_cost(solver);
    backward_pass_grad(solver);
    forward_pass(solver);
    update_slack(solver);
    update_dual(solver);
}

tinytype residual(const TinyWorkspace* w, const Point& old) {
    return std::sqrt((w->vnew - old.v).squaredNorm() +
                     (w->znew - old.z).squaredNorm() +
                     (w->g - old.g).squaredNorm() +
                     (w->y - old.y).squaredNorm());
}

TinySolver* make_solver(int max_evals) {
    static_assert(NHORIZON >= 1, "At least one control interval required");
    static_assert(NTOTAL >= NHORIZON + 1,
                  "Reference series is shorter than MPC horizon");

    // OSQP's NHORIZON counts control INTERVALS. TinyMPC counts state columns.
    const int N = NHORIZON + 1;
    const tinyMatrix A = Eigen::Map<const RowMajorMatrix>(Adyn_data,
                                                           NSTATES, NSTATES);
    const tinyMatrix B = Eigen::Map<const RowMajorMatrix>(Bdyn_data,
                                                           NSTATES, NINPUTS);
    const Vec Qdiag = Eigen::Map<const Vec>(Q_data, NSTATES);
    const Vec Rdiag = Eigen::Map<const Vec>(R_data, NINPUTS);

    TinySolver* s = nullptr;
    const int status = tiny_setup(&s, A, B, Vec::Zero(NSTATES),
                                  Qdiag.asDiagonal(), Rdiag.asDiagonal(),
                                  rho_value, NSTATES, NINPUTS, N, 0);
    if (status != 0 || !s) {
        std::cerr << "tiny_setup failed with status=" << status << "\n";
        return nullptr;
    }

    const Vec xmin_vec = Eigen::Map<const Vec>(xmin, NSTATES);
    const Vec xmax_vec = Eigen::Map<const Vec>(xmax, NSTATES);
    const Vec umin_vec = Eigen::Map<const Vec>(umin, NINPUTS);
    const Vec umax_vec = Eigen::Map<const Vec>(umax, NINPUTS);

    if (tiny_set_bound_constraints(
            s, xmin_vec.replicate(1, N), xmax_vec.replicate(1, N),
            umin_vec.replicate(1, N-1), umax_vec.replicate(1, N-1)) != 0) {
        std::cerr << "tiny_set_bound_constraints failed\n";
        return nullptr;
    }
    s->settings->adaptive_rho = 0;
    s->settings->en_state_bound = 1;
    s->settings->en_input_bound = 1;
    s->settings->max_iter = max_evals;
    s->settings->check_termination = 1;  // Required by reflected solver validation.

    // The generated Xref_data contains NTOTAL rows of NSTATES values.
    const auto refs = Eigen::Map<const RowMajorMatrix>(Xref_data,
                                                        NTOTAL, NSTATES);
    s->work->Xref = refs.topRows(N).transpose().eval();

    // gen_mpc_problem.py sets the stored initial state x0 to zero.
    if (tiny_set_x0(s, Vec::Zero(NSTATES)) != 0) {
        std::cerr << "tiny_set_x0 failed\n";
        return nullptr;
    }
    return s;
}

struct Trace {
    std::ofstream* out;
    std::string method;
    int first_below = -1;
    tinytype first_fp = -1;
    tinytype final_fp = -1;
    tinytype final_primal = -1;
    int restart_count = 0;
    int rows = 0;
};

constexpr tinytype TOL = 1e-4;

void record(TinySolver* s, Trace& ctx, int evaluation, int epoch,
            int epoch_step, int restarted, tinytype fp) {
    const TinyWorkspace* w = s->work;
    const tinytype primal = std::max(
        (w->x - w->vnew).cwiseAbs().maxCoeff(),
        (w->u - w->znew).cwiseAbs().maxCoeff());
    if (evaluation == 1) ctx.first_fp = fp;
    ctx.final_fp = fp;
    ctx.final_primal = primal;
    ++ctx.rows;
    if (restarted) ++ctx.restart_count;
    if (ctx.first_below < 0 && fp <= TOL) ctx.first_below = evaluation;
    (*ctx.out) << ctx.method << ',' << evaluation << ',' << fp << ','
               << primal << ',' << epoch << ',' << epoch_step << ','
               << restarted << '\n';
}

void reflected_callback(TinySolver* s, int eval, int epoch,
                        int epoch_step, int restart, tinytype fp,
                        void* opaque) {
    record(s, *static_cast<Trace*>(opaque), eval, epoch,
           epoch_step, restart, fp);
}

void trace_admm(TinySolver* s, Trace& ctx, int max_evals) {
    TinyWorkspace* w = s->work;
    for (int eval = 1; eval <= max_evals; ++eval) {
        const Point before = read_point(w);
        w->v = before.v;
        w->z = before.z;
        apply_raw_admm(s);
        const tinytype fp = residual(w, before);
        if (!std::isfinite(fp)) {
            std::cerr << "Nonfinite ADMM residual at evaluation " << eval << '\n';
            return;
        }
        record(s, ctx, eval, 0, eval-1, 0, fp);
    }
}

int positive_int(const char* arg) {
    char* end = nullptr;
    long v = std::strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || v < 1 || v > 1000000) return -1;
    return static_cast<int>(v);
}

void summary(const Trace& ctx) {
    std::cout << ctx.method << ": first_fp=" << ctx.first_fp
              << ", first_below_1e-4=" << ctx.first_below
              << ", final_fp=" << ctx.final_fp
              << ", final_primal_inf=" << ctx.final_primal
              << ", restarts=" << ctx.restart_count << '\n';
}

int main(int argc, char** argv) {
    const int K = argc > 1 ? positive_int(argv[1]) : 2;
    const int max_evals = argc > 2 ? positive_int(argv[2]) : 500;
    const std::string output = argc > 3 ? argv[3] : "result/random_mpc_fixed.csv";
    if (argc > 4 || K < 1 || max_evals < 1) {
        std::cerr << "Usage: random_mpc_fixed_benchmark [K] "
                     "[max_evals] [output.csv]\n";
        return 2;
    }
    std::ofstream out(output);
    if (!out) {
        std::cerr << "Cannot write " << output << ". Create its parent directory.\n";
        return 2;
    }
    out << std::setprecision(17);
    out << "method,evaluation,fp_residual,primal_inf,epoch,epoch_step,restart\n";

    TinySolver* admm = make_solver(max_evals);
    TinySolver* reflected = make_solver(max_evals);
    if (!admm || !reflected) return 2;
    Trace regular{&out, "admm"};
    Trace accelerated{&out, "reflected_fixed"};
    trace_admm(admm, regular, max_evals);
    const int rc = tiny_solve_halpern_reflected_fixed_restart_trace(
        reflected, K, tinytype(1.0), 0, reflected_callback, &accelerated);
    if (rc != 1) {
        std::cerr << "Reflected solver returned " << rc
                  << " (expected 1: max-evaluation budget exhausted).\n";
        return 2;
    }
    summary(regular);
    summary(accelerated);
    if (regular.rows != max_evals || accelerated.rows != max_evals) {
        std::cerr << "ERROR: missing residual rows\n";
        return 2;
    }
    if (std::abs(regular.first_fp-accelerated.first_fp) > 1e-10) {
        std::cerr << "ERROR: initial operator evaluations disagree\n";
        return 2;
    }
    std::cout << "Initial operator evaluations agree. Saved: " << output << '\n';
    return 0;
}
