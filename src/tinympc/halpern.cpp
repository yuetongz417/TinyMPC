#include "halpern.hpp"
#include "admm.hpp"

#include <iostream>

namespace {

// The state needed to determine the NEXT ADMM step. x/u are recomputed
// by the Riccati passes, and v/z (without 'new') are residual scratch.
struct HalpernAnchor {
    tinyMatrix v;  // saved work->vnew, nx x N
    tinyMatrix z;  // saved work->znew, nu x (N-1)
    tinyMatrix g;  // saved scaled state multiplier
    tinyMatrix y;  // saved scaled input multiplier
};

// Existing TinyMPC ADMM fixed-point map T, evaluated in place.
// The five routines are already declared in admm.hpp.
void apply_admm_operator(TinySolver* solver)
{
    update_linear_cost(solver);
    backward_pass_grad(solver);
    forward_pass(solver);
    update_slack(solver);
    update_dual(solver);
}

// Return the raw projected ADMM output, NOT the Halpern-averaged slack.
// This matches the existing TinyMPC solution convention.
void save_solution(TinySolver* solver, bool solved)
{
    solver->solution->iter = solver->work->iter;
    solver->solution->solved = solved ? 1 : 0;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;
    if (solved) {
        solver->work->status = 1;  // TINY_SOLVED
    }
}

}  // namespace

extern "C" int tiny_solve_halpern(TinySolver* solver)
{
    if (!solver || !solver->work || !solver->cache ||
        !solver->settings || !solver->solution) {
        std::cerr << "Halpern ADMM: invalid solver pointer.\n";
        return 2;
    }

    TinyWorkspace* w = solver->work;
    TinySettings* settings = solver->settings;

    if (settings->max_iter <= 0 || settings->check_termination <= 0 ||
        w->N < 2 || w->nx <= 0 || w->nu <= 0) {
        std::cerr << "Halpern ADMM: invalid dimensions or iteration settings.\n";
        return 2;
    }

    // Fixed rho is required to study one fixed-point map T.
    if (settings->adaptive_rho) {
        std::cerr << "Halpern ADMM: disable adaptive_rho first.\n";
        return 2;
    }

    // This first version averages only the four box-ADMM state arrays.
    // Additional constraints use additional slack/multiplier arrays.
    if (settings->en_state_soc || settings->en_input_soc ||
        settings->en_state_linear || settings->en_input_linear ||
        settings->en_tv_state_linear || settings->en_tv_input_linear) {
        std::cerr << "Halpern ADMM: only box constraints supported.\n";
        return 2;
    }

    solver->solution->iter = 0;
    solver->solution->solved = 0;
    w->status = 11;  // TINY_UNSOLVED
    w->iter = 0;

    // Capture w^0 ONCE. Do not update this anchor within the loop.
    const HalpernAnchor anchor{w->vnew, w->znew, w->g, w->y};

    for (int k = 0; k < settings->max_iter; ++k) {
        // The previous SLACK iterate is needed by termination_condition().
        // Here it is w^k, which may already be Halpern-averaged.
        w->v = w->vnew;
        w->z = w->znew;

        // Raw ADMM evaluation: w_hat = T(w^k).
        // x and u are the primal trajectory produced by this evaluation.
        apply_admm_operator(solver);
        w->iter = k + 1;

        // Check raw ADMM optimality residuals BEFORE Halpern averaging.
        // In particular, x-vnew is meaningful only with raw projected vnew.
        if (termination_condition(solver)) {
            save_solution(solver, true);
            std::cout << "Halpern ADMM converged in " << w->iter
                      << " iterations\n";
            return 0;
        }

        // Keep the raw projected trajectory if we have no next iteration.
        // An averaged slack need not satisfy the box constraints.
        if (k + 1 == settings->max_iter) {
            break;
        }

        // Halpern: w^{k+1} = alpha_k T(w^k) + beta_k w^0.
        // k starts at 0, so beta_0 = 1/2 and alpha_0 = 1/2.
        const tinytype beta = tinytype(1) / tinytype(k + 2);
        const tinytype alpha = tinytype(1) - beta;

        // 'vnew' and 'znew' are the slacks read by the next ADMM step.
        // 'v' and 'z' are not the fixed-point state variables here.
        w->vnew = alpha * w->vnew + beta * anchor.v;
        w->znew = alpha * w->znew + beta * anchor.z;
        w->g    = alpha * w->g    + beta * anchor.g;
        w->y    = alpha * w->y    + beta * anchor.y;
    }

    // The final operator evaluation remains unaveraged, so the returned
    // slacks are projected rather than a potentially infeasible average.
    save_solution(solver, false);
    return 1;  // Hit max_iter
}
