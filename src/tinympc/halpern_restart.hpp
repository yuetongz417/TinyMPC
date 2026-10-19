#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Called after evaluating raw T(w), before averaging or accepting a restart.
// total_eval: 1-based total number of ADMM operator evaluations.
// epoch: 0-based restart epoch.
// epoch_step: number of completed Halpern averages in this epoch before
//             this raw T evaluation (0 on its first evaluation).
// restart_now: 1 if this raw evaluation becomes the anchor of the next epoch.
// fp_residual: ||T(w)-w||_2 computed on (vnew,znew,g,y).
typedef void (*TinyRestartTraceCallback)(
    TinySolver* solver,
    int total_eval,
    int epoch,
    int epoch_step,
    int restart_now,
    tinytype fp_residual,
    void* user_data);

// Fixed-frequency restarted Halpern averaging of TinyMPC's box ADMM map.
// Each restart follows Algorithm 1 of Lu & Yang by taking one additional
// raw T step after `restart_every` Halpern updates.
// stop_on_convergence: 1 = use standard TinyMPC residual stopping;
//                      0 = run the full max_iter budget for diagnostics.
// Returns: 0 = converged, 1 = iteration budget reached, 2 = invalid settings.
int tiny_solve_halpern_fixed_restart_trace(
    TinySolver* solver,
    int restart_every,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data);

// Normal stopping behavior for use in closed-loop MPC.
int tiny_solve_halpern_fixed_restart(TinySolver* solver, int restart_every);

#ifdef __cplusplus
}
#endif
