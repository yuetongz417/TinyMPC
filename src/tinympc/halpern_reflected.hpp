#pragma once

// Use the same callback and solver types as the existing fixed-restart method.
#include "halpern_restart.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental application of Lu--Yang Algorithm 2 (reflected Halpern PDHG)
// to the BOX-CONSTRAINED TinyMPC ADMM map T on (vnew,znew,g,y).
//
// R_gamma(w) = w + 2*gamma*(T(w)-w),
// gamma = 1   -> full reflection 2T-I (paper Equation 26)
// gamma = 0.5 -> unreflected T (regression-control mode).
// Only gamma in [0.5,1] is accepted. This DOES NOT establish the paper's
// PDHG nonexpansiveness theorem for TinyMPC's ADMM representation.
//
// Every call to raw ADMM T counts as ONE operator evaluation.
// At a restart we use the already-computed *raw* T(w) as next anchor,
// NOT R_gamma(w). Residual callback measures ||T(w)-w||_2 BEFORE mixing.
// stop_on_convergence: 0 = full max_iter trace, 1 = standard stop test.
// Return: 0 = converged, 1 = budget exhausted, 2 = invalid/nonfinite.

int tiny_solve_halpern_reflected_fixed_restart_trace(
    TinySolver* solver,
    int restart_every,
    tinytype reflection_gamma,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data);

int tiny_solve_halpern_reflected_adaptive_restart_trace(
    TinySolver* solver,
    int first_epoch_steps,
    tinytype reflection_gamma,
    int stop_on_convergence,
    TinyRestartTraceCallback callback,
    void* user_data);

int tiny_solve_halpern_reflected_fixed_restart(
    TinySolver* solver, int restart_every, tinytype reflection_gamma);

int tiny_solve_halpern_reflected_adaptive_restart(
    TinySolver* solver, int first_epoch_steps, tinytype reflection_gamma);

#ifdef __cplusplus
}
#endif
