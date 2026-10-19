#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Solve the current MPC problem using Halpern averaging of TinyMPC's
// existing box-constraint ADMM operator.
//
// Returns 0 if the ordinary ADMM primal/dual residual test passes,
//         1 if max_iter is reached,
//         2 if the settings/inputs are unsupported or invalid.
//
// The ADMM penalty rho must remain fixed. No reflection or restart.
int tiny_solve_halpern(TinySolver* solver);

#ifdef __cplusplus
}
#endif
