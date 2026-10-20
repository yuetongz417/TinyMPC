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
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

using Vec = tinyVector;
using quadcopter::NX;
using quadcopter::NU;
using quadcopter::N;
using quadcopter::NH;

namespace {

constexpr tinytype FP_TOL = tinytype(1e-4);

// Tested scenarios; the singular variants remove strong convexity.
// Note: these are modified optimization problems, not the original OSQP example.
struct ProblemData {
    Vec Q, R, xr, xmin, xmax, umin, umax;
};

// Scenario names match the physical state interpretation in the OSQP A/B matrices:
// x(2) = yaw angle (its rate is x(8)); x(5) = height (its rate is x(11)).
// The tracking reference is yaw=1, height=0, not height=1.
ProblemData make_problem(const std::string& name) {
    ProblemData p;
    p.Q = quadcopter::q_diagonal();
    p.R = quadcopter::r_diagonal();
    p.xr = quadcopter::reference_state();
    quadcopter::state_bounds(p.xmin, p.xmax);
    quadcopter::input_bounds(p.umin, p.umax);
    if (name == "original") {
        // OSQP sample: Q original, R=0.1I, QN may differ inside TinyMPC.
    } else if (name == "yaw_cap_080") {
        p.xmax(2) = tinytype(0.8); // yaw target 1, ceiling 0.8
    } else if (name == "yaw_cap_020") {
        p.xmax(2) = tinytype(0.2);
    } else if (name == "yaw_height_singular" ||
               name == "yaw_height_low" ||
               name == "yaw_fixed_zero") {
        // Preserve OSQP dynamics and its tracking reference, but penalize
        // only yaw (index 2) and height (index 5). All inputs unpenalized.
        // The OSQP QP has P PSD and singular; condensed H is also singular.
        p.Q.setZero();
        p.Q(2) = tinytype(10.0);
        p.Q(5) = tinytype(10.0);
        p.R.setZero();
        if (name == "yaw_height_singular") {
            p.xmax(2) = tinytype(0.8);
        } else if (name == "yaw_height_low") {
            p.xmax(2) = tinytype(0.2);
        } else {
            p.xmin(2) = tinytype(0.0);
            p.xmax(2) = tinytype(0.0);
        }
    } else {
        throw std::invalid_argument("unknown scenario '" + name + "'");
    }
    return p;
}

// Eigenanalysis of the nominal OSQP QP with QN=Q and fixed x0=0.
// This checks genuine flat directions AFTER eliminating equality dynamics.
void print_osqp_hessian_spectrum(const ProblemData& p) {
    const tinyMatrix A = quadcopter::dynamics_A();
    const tinyMatrix B = quadcopter::dynamics_B();
    tinyMatrix S = tinyMatrix::Zero(NX * N, NU * NH);
    for (int j = 0; j < NH; ++j) {
        tinyMatrix Apower = tinyMatrix::Identity(NX, NX);
        for (int t = j + 1; t < N; ++t) {
            S.block(t * NX, j * NU, NX, NU) = Apower * B;
            Apower = A * Apower;
        }
    }
    const Vec q_stacked = p.Q.replicate(N, 1);
    const Vec r_stacked = p.R.replicate(NH, 1);
    tinyMatrix H = (S.transpose() * q_stacked.asDiagonal() * S).eval();
    H.diagonal() += r_stacked;
    Eigen::SelfAdjointEigenSolver<tinyMatrix> solver(H);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition of condensed H failed");
    }
    const Vec eig = solver.eigenvalues();
    const double threshold = std::max(1e-8, 1e-10 * std::abs(double(eig(eig.size()-1))));
    int near_zero = 0;
    for (int i=0; i<eig.size(); ++i) if (std::abs(double(eig(i)))<threshold) ++near_zero;
    int full_zeros = 0;
    // OSQP QN=Q: N+1 state-cost copies, and N input-cost copies.
    // N = 11 state stages already includes the terminal state exactly once.
    for (int i=0;i<NX;++i) if (std::abs(double(p.Q(i)))<1e-12) full_zeros += N;
    for (int j=0;j<NU;++j) if (std::abs(double(p.R(j)))<1e-12) full_zeros += NH;
    std::cout << "Nominal OSQP full P (" << NX*N + NU*NH << "x"
              << NX*N + NU*NH << "): zero eigenvalue count = " << full_zeros << '\n';
    std::cout << "OSQP condensed H (" << NU*NH << "x" << NU*NH << ")"
              << ": lambda_min=" << eig(0)
              << ", lambda_max=" << eig(eig.size()-1)
              << ", near-zero eigenvalues=" << near_zero << '\n';
    std::cout << "NOTE: TinyMPC's Riccati terminal cost can differ from OSQP's QN=Q.\n";
}


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

TinySolver* make_solver(int max_evals, tinytype rho, const ProblemData& p) {
    const tinyMatrix A = quadcopter::dynamics_A();
    const tinyMatrix B = quadcopter::dynamics_B();

    TinySolver* s = nullptr;
    const int status = tiny_setup(&s, A, B, Vec::Zero(NX),
                                  p.Q.asDiagonal(), p.R.asDiagonal(),
                                  rho, NX, NU, N, 0);
    if (status != 0 || !s) {
        std::cerr << "tiny_setup failed with status=" << status << '\n';
        return nullptr;
    }

    if (tiny_set_bound_constraints(s,
            p.xmin.replicate(1, N), p.xmax.replicate(1, N),
            p.umin.replicate(1, NH), p.umax.replicate(1, NH)) != 0) {
        std::cerr << "tiny_set_bound_constraints failed\n";
        return nullptr;
    }

    s->settings->adaptive_rho = 0;   // Keep original operator T unchanged.
    s->settings->en_state_bound = 1;
    s->settings->en_input_bound = 1;
    s->settings->max_iter = max_evals;
    s->settings->check_termination = 1;

    if (tiny_set_x_ref(s, p.xr.replicate(1, N)) != 0) {
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

Diagnostics diagnostics(const TinyWorkspace* w, const ProblemData& p) {
    Diagnostics d;
    const tinyMatrix A = quadcopter::dynamics_A();
    const tinyMatrix B = quadcopter::dynamics_B();
    const Vec& Qdiag = p.Q;
    const Vec& Rdiag = p.R;
    const Vec& xr = p.xr;
    const Vec& xmin = p.xmin;
    const Vec& xmax = p.xmax;
    const Vec& umin = p.umin;
    const Vec& umax = p.umax;

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
    ProblemData problem;
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
    const Diagnostics d = diagnostics(w,ctx.problem);
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
    const std::string output = argc > 3 ? argv[3] : "quadcopter_modified.csv";
    const tinytype rho = argc > 4 ? positive_rho(argv[4]) : tinytype(1.0);
    const std::string scenario = argc > 5 ? argv[5] : "original";
    const tinytype gamma = argc > 6 ? positive_rho(argv[6]) : tinytype(1.0);
    if (argc > 7 || K < 1 || max_evals < 1 ||
        !std::isfinite(rho) || rho <= 0 ||
        !std::isfinite(gamma) || gamma < tinytype(0.5) || gamma > tinytype(1)) {
        std::cerr << "Usage: quadcopter_hardcase_benchmark [K] [max_evals] "
                     "[output.csv] [rho] [scenario] [gamma]\n";
        return 2;
    }
    ProblemData p;
    try {
        p=make_problem(scenario);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
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
    print_osqp_hessian_spectrum(p);
    std::cout << "Quadcopter first-window MPC: nx=" << NX << " nu=" << NU
              << " Nh=" << NH << " N_tinympc=" << N
              << " scenario=" << scenario << " K=" << K
              << " gamma=" << gamma << " rho=" << rho
              << " evaluations=" << max_evals << '\n';

    TinySolver* admm = make_solver(max_evals, rho, p);
    TinySolver* reflected = make_solver(max_evals, rho, p);
    if (!admm || !reflected) return 2;

    Trace regular{&out, "admm", p};
    Trace accelerated{&out, "reflected_fixed", p};
    trace_admm(admm, regular, max_evals);
    const int rc = tiny_solve_halpern_reflected_fixed_restart_trace(
        reflected, K, gamma, 0, reflected_callback, &accelerated);
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
    std::cout << "NOTE: tracking_cost_qn_eq_q is a Q/Q_N=Q tracking "
                 "diagnostic with the modified state/control weights; "
                 "it is not necessarily TinyMPC's terminal-cost objective.\n";
    return 0;
}
