#pragma once

// Reuse the exact trace callback type already used by the existing
// fixed-frequency restart solver.
#include "halpern_restart.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Paper-style adaptive restart of TinyMPC's BOX-CONSTRAINED ADMM operator.
// first_epoch_steps: number of Halpern averages in epoch 0 before the first
//     restart's extra raw T evaluation (2 -> first restart at evaluation 3).
// Subsequent epochs restart when ||T(w)-w|| <= exp(-1)*||T(anchor)-anchor||,
//     requiring at least one completed Halpern update in that epoch.
// stop_on_convergence: 0 = trace full max_iter budget; 1 = TinyMPC stop test.
// callback: same signature and timing as fixed restart: after raw T(w),
//           before accepting the restart or Halpern averaging.
// Operator evaluations (including restart evaluations) count toward max_iter.
// Returns: 0 = converged; 1 = max_iter reached; 2 = invalid/unsupported inputs.
int tiny_solve_halpern_adaptive_restart_trace(
    TinySolver* solver,
    int first_epoch_steps,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data);

// Production version using the standard TinyMPC primal/dual stopping rule.
int tiny_solve_halpern_adaptive_restart(
    TinySolver* solver,
    int first_epoch_steps);

#ifdef __cplusplus
}
#endif
