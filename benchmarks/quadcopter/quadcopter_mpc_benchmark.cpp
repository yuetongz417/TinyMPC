// One fixed quadcopter MPC QP: ordinary ADMM vs reflected fixed-restart Halpern.
// Data: 12-state, 4-input OSQP quadcopter example (N_h = 10 intervals).
// Usage: quadcopter_mpc_benchmark [K=2] [max_evals=500] [output.csv] [rho=1]
// Traces count evaluations of the SAME original TinyMPC ADMM operator T.
// NOTE: native TinyMPC terminal treatment can differ from OSQP's QN=Q.

#include <tinympc/tiny_api.hpp>
#include <tinympc/admm.hpp>
#include <tinympc/halpern_reflected.hpp>
#include "quadcopter_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using Vec = tinyVector;
using quadcopter::NX;
using quadcopter::NU;
using quadcopter::N;
using quadcopter::NH;

namespace {

constexpr tinytype FP_TOL = tinytype(1e-4);

struct Point { tinyMatrix v, z, g, y; };

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

tinytype fixed_point_residual(const TinyWorkspace* w, const Point& before) {
    return std::sqrt((w->vnew - before.v).squaredNorm() +
                     (w->znew - before.z).squaredNorm() +
                     (w->g - before.g).squaredNorm() +
                     (w->y - before.y).squaredNorm());
}

TinySolver* make_solver(int max_evals, tinytype rho) {
    const tinyMatrix A = quadcopter::dynamics_A();
    const tinyMatrix B = quadcopter::dynamics_B();
    const Vec Qdiag = quadcopter::q_diagonal();
    const Vec Rdiag = quadcopter::r_diagonal();

    TinySolver* s = nullptr;
    const int status = tiny_setup(&s, A, B, Vec::Zero(NX),
                                  Qdiag.asDiagonal(), Rdiag.asDiagonal(),
                                  rho, NX, NU, N, 0);
    if (status != 0 || !s) {
        std::cerr << "tiny_setup failed with status=" << status << '\n';
        return nullptr;
    }

    Vec xmin, xmax, umin, umax;
    quadcopter::state_bounds(xmin, xmax);
    quadcopter::input_bounds(umin, umax);
    if (tiny_set_bound_constraints(s,
            xmin.replicate(1, N), xmax.replicate(1, N),
            umin.replicate(1, NH), umax.replicate(1, NH)) != 0) {
        std::cerr << "tiny_set_bound_constraints failed\n";
        return nullptr;
    }

    s->settings->adaptive_rho = 0;   // Keep original operator T unchanged.
    s->settings->en_state_bound = 1;
    s->settings->en_input_bound = 1;
    s->settings->max_iter = max_evals;
    s->settings->check_termination = 1;

    const Vec xr = quadcopter::reference_state();
    if (tiny_set_x_ref(s, xr.replicate(1, N)) != 0) {
        std::cerr << "tiny_set_x_ref failed\n";
        return nullptr;
    }
    if (tiny_set_u_ref(s, tinyMatrix::Zero(NU, NH)) != 0) {
        std::cerr << "tiny_set_u_ref failed\n";
        return nullptr;
    }
    // FIRST QP ONLY: x0 = 0, as in the first OSQP closed-loop solve.
    if (tiny_set_x0(s, Vec::Zero(NX)) != 0) {
        std::cerr << "tiny_set_x0 failed\n";
        return nullptr;
    }
    return s;
}

// The following diagnostics inspect the RAW trajectory T(w) produced by the
// operator evaluation. They are not necessarily the accepted extrapolated
// Halpern iterate (the callback runs before the Halpern averaging/restart).
struct Diagnostics {
    tinytype state_bound_inf = 0;
    tinytype input_bound_inf = 0;
    tinytype dynamics_inf = 0;
    tinytype tracking_cost_qn_eq_q = 0;
    tinytype max_abs_state = 0;
};

Diagnostics diagnostics(const TinyWorkspace* w) {
    Diagnostics d;
    const tinyMatrix A = quadcopter::dynamics_A();
    const tinyMatrix B = quadcopter::dynamics_B();
    const Vec Qdiag = quadcopter::q_diagonal();
    const Vec Rdiag = quadcopter::r_diagonal();
    const Vec xr = quadcopter::reference_state();
    Vec xmin, xmax, umin, umax;
    quadcopter::state_bounds(xmin, xmax);
    quadcopter::input_bounds(umin, umax);

    for (int t = 0; t < N; ++t) {
        const Vec x = w->x.col(t);
        d.max_abs_state = std::max(d.max_abs_state, x.cwiseAbs().maxCoeff());
        d.state_bound_inf = std::max(d.state_bound_inf,
            std::max(tinytype(0),
                std::max((xmin - x).maxCoeff(), (x - xmax).maxCoeff())));
        const Vec dx = x - xr;
        d.tracking_cost_qn_eq_q += tinytype(0.5) *
            (Qdiag.array() * dx.array().square()).sum();
        if (t < NH) {
            const Vec u = w->u.col(t);
            d.input_bound_inf = std::max(d.input_bound_inf,
                std::max(tinytype(0),
                    std::max((umin - u).maxCoeff(), (u - umax).maxCoeff())));
            d.tracking_cost_qn_eq_q += tinytype(0.5) *
                (Rdiag.array() * u.array().square()).sum();
            const Vec defect = w->x.col(t + 1) - A*x - B*u;
            d.dynamics_inf = std::max(d.dynamics_inf,
                                      defect.cwiseAbs().maxCoeff());
        }
    }
    return d;
}

struct Trace {
    std::ofstream* out;
    std::string method;
    int first_below = -1;
    tinytype first_fp = -1;
    tinytype final_fp = -1;
    tinytype final_primal = -1;
    tinytype final_input_violation = -1;
    int restart_count = 0;
    int rows = 0;
};

void record(TinySolver* s, Trace& ctx, int eval, int epoch,
            int epoch_step, int restart, tinytype fp) {
    const TinyWorkspace* w = s->work;
    const tinytype primal_inf = std::max(
        (w->x - w->vnew).cwiseAbs().maxCoeff(),
        (w->u - w->znew).cwiseAbs().maxCoeff());
    const Diagnostics d = diagnostics(w);
    if (eval == 1) ctx.first_fp = fp;
    ctx.final_fp = fp;
    ctx.final_primal = primal_inf;
    ctx.final_input_violation = d.input_bound_inf;
    ++ctx.rows;
    if (restart) ++ctx.restart_count;
    if (ctx.first_below < 0 && fp <= FP_TOL) ctx.first_below = eval;
    (*ctx.out) << ctx.method << ',' << eval << ',' << fp << ','
               << primal_inf << ',' << epoch << ',' << epoch_step << ','
               << restart << ',' << d.state_bound_inf << ','
               << d.input_bound_inf << ',' << d.dynamics_inf << ','
               << d.tracking_cost_qn_eq_q << ',' << d.max_abs_state << '\n';
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
        const tinytype fp = fixed_point_residual(w, before);
        if (!std::isfinite(fp)) {
            std::cerr << "Nonfinite ADMM fixed-point residual at eval "
                      << eval << '\n';
            return;
        }
        record(s, ctx, eval, 0, eval - 1, 0, fp);
    }
}

int positive_int(const char* arg) {
    char* end = nullptr;
    const long v = std::strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || v < 1 || v > 1000000) return -1;
    return static_cast<int>(v);
}

tinytype positive_rho(const char* arg) {
    char* end = nullptr;
    const double v = std::strtod(arg, &end);
    if (end == arg || *end != '\0' || !std::isfinite(v) || v <= 0)
        return tinytype(-1);
    return tinytype(v);
}

void print_summary(const Trace& t) {
    std::cout << t.method << ": first_fp=" << t.first_fp
              << ", first_below_1e-4=" << t.first_below
              << ", final_fp=" << t.final_fp
              << ", final_primal_inf=" << t.final_primal
              << ", final_input_bound_inf=" << t.final_input_violation
              << ", restarts=" << t.restart_count << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const int K = argc > 1 ? positive_int(argv[1]) : 2;
    const int max_evals = argc > 2 ? positive_int(argv[2]) : 500;
    const std::string output = argc > 3 ? argv[3] : "quadcopter_K2.csv";
    const tinytype rho = argc > 4 ? positive_rho(argv[4]) : tinytype(1.0);
    if (argc > 5 || K < 1 || max_evals < 1 ||
        !std::isfinite(rho) || rho <= 0) {
        std::cerr << "Usage: quadcopter_mpc_benchmark [K] [max_evals] "
                     "[output.csv] [rho]\n";
        return 2;
    }
    std::ofstream out(output);
    if (!out) {
        std::cerr << "Cannot write '" << output
                  << "'. Create its parent directory first.\n";
        return 2;
    }
    out << std::setprecision(17);
    out << "method,evaluation,fp_residual,primal_inf,epoch,epoch_step,restart,"
           "state_bound_inf,input_bound_inf,dynamics_inf,"
           "tracking_cost_qn_eq_q,max_abs_state\n";
    std::cout << "Quadcopter first-window MPC: nx=" << NX << " nu=" << NU
              << " Nh=" << NH << " N_tinympc=" << N
              << " K=" << K << " gamma=1 rho=" << rho
              << " evaluations=" << max_evals << '\n';

    TinySolver* admm = make_solver(max_evals, rho);
    TinySolver* reflected = make_solver(max_evals, rho);
    if (!admm || !reflected) return 2;

    Trace regular{&out, "admm"};
    Trace accelerated{&out, "reflected_fixed"};
    trace_admm(admm, regular, max_evals);
    const int rc = tiny_solve_halpern_reflected_fixed_restart_trace(
        reflected, K, tinytype(1.0), 0, reflected_callback, &accelerated);
    if (rc != 1) {
        std::cerr << "Reflected solver returned " << rc
                  << " (expected 1: evaluation budget exhausted).\n";
        return 2;
    }
    print_summary(regular);
    print_summary(accelerated);
    if (regular.rows != max_evals || accelerated.rows != max_evals) {
        std::cerr << "ERROR: incomplete trace. Check for nonfinite residuals.\n";
        return 2;
    }
    if (std::abs(regular.first_fp - accelerated.first_fp) > tinytype(1e-10)) {
        std::cerr << "ERROR: first operator evaluations do not agree.\n";
        return 2;
    }
    std::cout << "Identical initial raw evaluation verified. CSV: "
              << output << '\n';
    std::cout << "NOTE: tracking_cost_qn_eq_q is the original OSQP tracking "
                 "cost diagnostic, not necessarily TinyMPC's actual "
                 "terminal-cost objective.\n";
    return 0;
}
