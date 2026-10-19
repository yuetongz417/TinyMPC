#include "halpern_adaptive.hpp"
#include "admm.hpp"

#include <cmath>
#include <iostream>

namespace {

// Only the data that determines the NEXT original ADMM operator evaluation.
// TinyMPC's primal x,u trajectories are reconstructed by the Riccati passes.
struct ADMMPoint {
    tinyMatrix v;   // work->vnew
    tinyMatrix z;   // work->znew
    tinyMatrix g;   // scaled state multiplier
    tinyMatrix y;   // scaled input multiplier
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
    // The latest raw slacks are feasible w.r.t. the box projection.
    // This matches the existing halpern_restart.cpp return convention.
    solver->solution->x = w->vnew;
    solver->solution->u = w->znew;
    w->status = converged ? 1 : 11;
}

bool valid(const TinySolver* solver, int first_epoch_steps) {
    if (!solver || !solver->work || !solver->cache ||
        !solver->settings || !solver->solution) return false;
    const TinyWorkspace* w = solver->work;
    const TinySettings* s = solver->settings;
    if (first_epoch_steps < 1 || s->max_iter < 1 ||
        s->check_termination < 1 || w->N < 2 || w->nx < 1 || w->nu < 1 ||
        !std::isfinite(solver->cache->rho) || solver->cache->rho <= 0)
        return false;
    // Keep T fixed within a solve; exactly the same constraints as
    // the fixed restart implementation supplied previously.
    if (s->adaptive_rho ||
        s->en_state_soc || s->en_input_soc ||
        s->en_state_linear || s->en_input_linear ||
        s->en_tv_state_linear || s->en_tv_input_linear)
        return false;
    return true;
}

}  // namespace

extern "C" int tiny_solve_halpern_adaptive_restart_trace(
    TinySolver* solver,
    int first_epoch_steps,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data)
{
    if (!valid(solver, first_epoch_steps)) {
        std::cerr << "Adaptive-restart Halpern: invalid settings or "
                     "unsupported constraints. Disable adaptive_rho.\n";
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
    int epoch_step = 0; // number of completed Halpern averages in epoch
    tinytype epoch_start_residual = tinytype(0);
    const tinytype reduction = std::exp(tinytype(-1));

    for (int eval = 1; eval <= s->max_iter; ++eval) {
        // current point is the previously accepted ADMM or Halpern state
        const ADMMPoint before = read_point(w);

        // TinyMPC uses v,z as scratch for old slacks in its stopping test.
        w->v = before.v;
        w->z = before.z;

        raw_admm_step(solver);          // raw = T(current)
        w->iter = eval;
        const tinytype fp = fixed_point_residual(w, before);

        if (!std::isfinite(fp)) {
            std::cerr << "Adaptive-restart Halpern: nonfinite residual at "
                      << "evaluation " << eval << "\n";
            save_solution(solver, false);
            return 2;
        }

        // At k=0, 'before' is exactly the epoch anchor. Consequently,
        // fp = ||T(anchor)-anchor|| is the denominator in Eq. (10).
        if (epoch_step == 0) {
            epoch_start_residual = fp;
        }

        // Preserve the production solver's ordinary termination test;
        // diagnostics disable it to trace the full evaluation budget.
        const bool converged =
            (stop_on_convergence != 0) && termination_condition(solver);
        const bool last_eval = (eval == s->max_iter);

        bool restart_now = false;
        if (!converged && !last_eval) {
            if (epoch == 0) {
                // First epoch: exactly first_epoch_steps Halpern
                // averages, then one EXTRA raw T step to restart.
                restart_now = (epoch_step == first_epoch_steps);
            } else {
                // Paper Eq. (10): r_{n,k} <= exp(-1)*r_{n,0}.
                // k>0 avoids restarting immediately at the anchor.
                restart_now = (epoch_step > 0) &&
                    (fp <= reduction * epoch_start_residual);
            }
        }

        // Logging is always on raw T(current), before averaging/restart.
        if (callback) {
            callback(solver, eval, epoch, epoch_step,
                     restart_now ? 1 : 0, fp, user_data);
        }

        if (converged) {
            save_solution(solver, true);
            return 0;
        }
        if (last_eval) {
            save_solution(solver, false);
            return 1;
        }

        if (restart_now) {
            // Paper's restart point a_{n+1}=T(w^{n,k}): it is ALREADY
            // available from the raw call above, no extra call needed.
            anchor = read_point(w);
            ++epoch;
            epoch_step = 0;
        } else {
            // Halpern update (k is LOCAL to the current restart epoch).
            const tinytype beta = tinytype(1) / tinytype(epoch_step + 2);
            const tinytype alpha = tinytype(1) - beta;
            w->vnew = alpha * w->vnew + beta * anchor.v;
            w->znew = alpha * w->znew + beta * anchor.z;
            w->g    = alpha * w->g    + beta * anchor.g;
            w->y    = alpha * w->y    + beta * anchor.y;
            ++epoch_step;
        }
    }
    return 1;  // Unreachable if s->max_iter >= 1.
}

extern "C" int tiny_solve_halpern_adaptive_restart(
    TinySolver* solver, int first_epoch_steps)
{
    return tiny_solve_halpern_adaptive_restart_trace(
        solver, first_epoch_steps, 1, nullptr, nullptr);
}
