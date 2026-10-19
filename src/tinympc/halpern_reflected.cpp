#include "halpern_reflected.hpp"
#include "admm.hpp"

#include <cmath>
#include <iostream>

namespace {

struct ADMMPoint {
    tinyMatrix v; // accepted state slack (work->vnew)
    tinyMatrix z; // accepted input slack (work->znew)
    tinyMatrix g; // state scaled multiplier
    tinyMatrix y; // input scaled multiplier
};

ADMMPoint get_point(const TinyWorkspace* w) {
    return {w->vnew, w->znew, w->g, w->y};
}

tinytype fp_residual(const TinyWorkspace* w, const ADMMPoint& before) {
    return std::sqrt(
        (w->vnew - before.v).squaredNorm() +
        (w->znew - before.z).squaredNorm() +
        (w->g - before.g).squaredNorm() +
        (w->y - before.y).squaredNorm());
}

void apply_raw_T(TinySolver* solver) {
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
    // The final iterate returned is the raw projected T(w), not the
    // extrapolated point, whose slacks need not satisfy box bounds.
    solver->solution->x = w->vnew;
    solver->solution->u = w->znew;
    w->status = converged ? 1 : 11;
}

bool valid(const TinySolver* solver, int steps, tinytype gamma) {
    if (!solver || !solver->work || !solver->cache ||
        !solver->settings || !solver->solution) return false;
    const TinyWorkspace* w = solver->work;
    const TinySettings* s = solver->settings;
    if (steps < 1 || s->max_iter < 1 || s->check_termination < 1 ||
        w->N < 2 || w->nx < 1 || w->nu < 1 ||
        !std::isfinite(gamma) || gamma < tinytype(0.5) || gamma > tinytype(1) ||
        !std::isfinite(solver->cache->rho) || solver->cache->rho <= 0)
        return false;

    // A fixed box ADMM operator. Extending this to SOC/linear constraints
    // requires reflecting their auxiliary variables and multipliers too.
    if (s->adaptive_rho || s->en_state_soc || s->en_input_soc ||
        s->en_state_linear || s->en_input_linear ||
        s->en_tv_state_linear || s->en_tv_input_linear)
        return false;
    return true;
}

enum class RestartMode { Fixed, Adaptive };

int solve(TinySolver* solver,
          int steps,
          tinytype gamma,
          RestartMode mode,
          int stop_on_convergence,
          TinyRestartTraceCallback callback,
          void* user_data) {
    if (!valid(solver, steps, gamma)) {
        std::cerr << "Reflected Halpern: invalid settings or unsupported "
                  << "constraints. Require fixed rho, box-only, "
                  << "gamma in [0.5,1].\n";
        return 2;
    }

    TinyWorkspace* w = solver->work;
    TinySettings* s = solver->settings;
    w->iter = 0;
    w->status = 11;
    solver->solution->iter = 0;
    solver->solution->solved = 0;

    ADMMPoint anchor = get_point(w);
    int epoch = 0;
    int epoch_step = 0; // number of *completed* Halpern updates
    tinytype epoch_start_residual = 0;
    const tinytype reduction = std::exp(tinytype(-1));

    for (int eval = 1; eval <= s->max_iter; ++eval) {
        const ADMMPoint before = get_point(w);

        // Keep TinyMPC's original stopping-test scratch variables valid.
        w->v = before.v;
        w->z = before.z;

        // ONE original, UNREFLECTED ADMM evaluation: raw = T(before).
        apply_raw_T(solver);
        w->iter = eval;
        const tinytype fp = fp_residual(w, before);
        if (!std::isfinite(fp)) {
            std::cerr << "Reflected Halpern: nonfinite ||T(w)-w|| at eval "
                      << eval << "\n";
            save_solution(solver, false);
            return 2;
        }

        // In the first raw evaluation of each epoch, 'before' is anchor.
        if (epoch_step == 0) epoch_start_residual = fp;

        const bool converged = (stop_on_convergence != 0) &&
                               termination_condition(solver);
        const bool last_eval = (eval == s->max_iter);

        bool restart_now = false;
        if (!converged && !last_eval) {
            if (mode == RestartMode::Fixed || epoch == 0) {
                // First epoch of adaptive follows same fixed-step protocol.
                // 2 updates -> raw restart T at evaluation 3.
                restart_now = (epoch_step == steps);
            } else {
                // Lu--Yang adaptive threshold; residual of reflected map
                // is 2*gamma*fp, so its ratio is identical to raw T ratio.
                restart_now = (epoch_step > 0) &&
                              (fp <= reduction * epoch_start_residual);
            }
        }

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
            // Paper Algorithm 2: a_next = T(w), NOT 2*T(w)-w.
            // This raw result has already been computed by apply_raw_T.
            anchor = get_point(w);
            ++epoch;
            epoch_step = 0;
        } else {
            const tinytype beta = tinytype(1) / tinytype(epoch_step + 2);
            const tinytype alpha = tinytype(1) - beta;
            const tinytype displacement_scale = tinytype(2) * gamma;

            // Lu--Yang Eq. 26 with gamma=1:
            // next = alpha * (2*T(before)-before) + beta * anchor.
            // gamma=0.5 exactly recovers the unreflected Halpern scheme.
            // Previous iterates are deep-copied into 'before', so these
            // assignments do not mix old and new components of the map.
            w->vnew = alpha * (before.v + displacement_scale *
                            (w->vnew - before.v)) + beta * anchor.v;
            w->znew = alpha * (before.z + displacement_scale *
                            (w->znew - before.z)) + beta * anchor.z;
            w->g    = alpha * (before.g + displacement_scale *
                            (w->g - before.g)) + beta * anchor.g;
            w->y    = alpha * (before.y + displacement_scale *
                            (w->y - before.y)) + beta * anchor.y;
            ++epoch_step;
        }
    }
    return 1;
}
} // namespace

extern "C" int tiny_solve_halpern_reflected_fixed_restart_trace(
    TinySolver* solver, int restart_every, tinytype reflection_gamma,
    int stop_on_convergence, TinyRestartTraceCallback callback, void* user_data) {
    return solve(solver, restart_every, reflection_gamma, RestartMode::Fixed,
                 stop_on_convergence, callback, user_data);
}

extern "C" int tiny_solve_halpern_reflected_adaptive_restart_trace(
    TinySolver* solver, int first_epoch_steps, tinytype reflection_gamma,
    int stop_on_convergence, TinyRestartTraceCallback callback, void* user_data) {
    return solve(solver, first_epoch_steps, reflection_gamma, RestartMode::Adaptive,
                 stop_on_convergence, callback, user_data);
}

extern "C" int tiny_solve_halpern_reflected_fixed_restart(
    TinySolver* solver, int restart_every, tinytype reflection_gamma) {
    return tiny_solve_halpern_reflected_fixed_restart_trace(
        solver, restart_every, reflection_gamma, 1, nullptr, nullptr);
}

extern "C" int tiny_solve_halpern_reflected_adaptive_restart(
    TinySolver* solver, int first_epoch_steps, tinytype reflection_gamma) {
    return tiny_solve_halpern_reflected_adaptive_restart_trace(
        solver, first_epoch_steps, reflection_gamma, 1, nullptr, nullptr);
}
