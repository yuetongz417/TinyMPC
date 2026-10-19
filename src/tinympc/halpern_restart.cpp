#include "halpern_restart.hpp"
#include "admm.hpp"

#include <cmath>
#include <iostream>

namespace {

// Exactly the variables read by the next box-constrained ADMM step.
// The x/u trajectories are recomputed by the Riccati passes.
struct ADMMPoint {
    tinyMatrix v;  // workspace vnew (nx by N)
    tinyMatrix z;  // workspace znew (nu by N-1)
    tinyMatrix g;  // scaled state multiplier
    tinyMatrix y;  // scaled input multiplier
};

ADMMPoint read_point(const TinyWorkspace* w) {
    return {w->vnew, w->znew, w->g, w->y};
}

tinytype fixed_point_residual(const TinyWorkspace* w,
                              const ADMMPoint& before) {
    return std::sqrt(
        (w->vnew - before.v).squaredNorm() +
        (w->znew - before.z).squaredNorm() +
        (w->g    - before.g).squaredNorm() +
        (w->y    - before.y).squaredNorm());
}

void raw_admm_step(TinySolver* solver) {
    update_linear_cost(solver);
    backward_pass_grad(solver);
    forward_pass(solver);
    update_slack(solver);
    update_dual(solver);
}

void save_solution(TinySolver* solver, bool converged) {
    TinyWorkspace* w = solver->work;
    solver->solution->iter = w->iter;
    solver->solution->solved = converged ? 1 : 0;
    // At the stopping point w->vnew and w->znew are raw projected slacks,
    // not potentially infeasible Halpern averages.
    solver->solution->x = w->vnew;
    solver->solution->u = w->znew;
    w->status = converged ? 1 : 11;  // TinyMPC solved / unsolved
}

bool valid(const TinySolver* solver, int K) {
    if (!solver || !solver->work || !solver->cache ||
        !solver->settings || !solver->solution) return false;
    const TinyWorkspace* w = solver->work;
    const TinySettings* s = solver->settings;
    if (K < 1 || s->max_iter < 1 || s->check_termination < 1 ||
        w->N < 2 || w->nx < 1 || w->nu < 1 ||
        !std::isfinite(solver->cache->rho) || solver->cache->rho <= 0)
        return false;
    // The operator and its fixed point must remain unchanged throughout
    // an inner solve; only box constraints are supported in this version.
    if (s->adaptive_rho ||
        s->en_state_soc || s->en_input_soc ||
        s->en_state_linear || s->en_input_linear ||
        s->en_tv_state_linear || s->en_tv_input_linear) return false;
    return true;
}

}  // namespace

extern "C" int tiny_solve_halpern_fixed_restart_trace(
    TinySolver* solver,
    int restart_every,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data)
{
    if (!valid(solver, restart_every)) {
        std::cerr << "Fixed-restart Halpern: invalid settings or unsupported "
                     "constraint type. Disable adaptive_rho.\n";
        return 2;
    }

    TinyWorkspace* w = solver->work;
    TinySettings* s = solver->settings;
    w->iter = 0;
    w->status = 11;
    solver->solution->iter = 0;
    solver->solution->solved = 0;

    ADMMPoint anchor = read_point(w);
    int epoch = 0;
    int epoch_step = 0;  // Number of Halpern steps completed this epoch.

    // The global budget counts every raw T evaluation, including the
    // special raw T step that begins the next epoch.
    for (int eval = 1; eval <= s->max_iter; ++eval) {
        const ADMMPoint before = read_point(w);

        // Keep TinyMPC's previous-slack scratch consistent for its normal
        // primal/dual stopping criterion (before raw projection).
        w->v = before.v;
        w->z = before.z;

        // raw = T(current)
        raw_admm_step(solver);
        w->iter = eval;
        const tinytype fp = fixed_point_residual(w, before);

        const bool converged = stop_on_convergence && termination_condition(solver);
        const bool last_eval = (eval == s->max_iter);

        // After K completed Halpern steps, this evaluation is the ONE
        // additional T(w^{n,K}) used to start the next restart epoch.
        // We do not restart on the final budget step or after convergence.
        const bool restart_now =
            !converged && !last_eval && (epoch_step == restart_every);

        if (callback)
            callback(solver, eval, epoch, epoch_step,
                     restart_now ? 1 : 0, fp, user_data);

        if (converged) {
            save_solution(solver, true);
            return 0;
        }
        if (last_eval) {
            save_solution(solver, false);
            return 1;
        }

        if (restart_now) {
            // New anchor is the raw T result (not the preceding Halpern
            // averaged iterate). No extra T call is necessary: this IS it.
            anchor = read_point(w);
            ++epoch;
            epoch_step = 0;
        } else {
            // One Halpern step inside the current epoch:
            // w^{n,k+1} = ((k+1)/(k+2)) T(w^{n,k}) + (1/(k+2)) a_n.
            const tinytype beta = tinytype(1) / tinytype(epoch_step + 2);
            const tinytype alpha = tinytype(1) - beta;
            w->vnew = alpha * w->vnew + beta * anchor.v;
            w->znew = alpha * w->znew + beta * anchor.z;
            w->g    = alpha * w->g    + beta * anchor.g;
            w->y    = alpha * w->y    + beta * anchor.y;
            ++epoch_step;
        }
    }
    return 1; // Unreachable for positive max_iter.
}

extern "C" int tiny_solve_halpern_fixed_restart(
    TinySolver* solver, int restart_every)
{
    return tiny_solve_halpern_fixed_restart_trace(
        solver, restart_every, 1, nullptr, nullptr);
}
