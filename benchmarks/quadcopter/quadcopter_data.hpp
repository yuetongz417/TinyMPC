#pragma once

// Data copied from the 12-state / 4-input OSQP quadcopter MPC example.
// State order, input order, and matrix entries are preserved.
// The original example uses N=10 *control intervals*.

#include <tinympc/types.hpp>
#include <cmath>
#include <limits>

namespace quadcopter {

constexpr int NX = 12;
constexpr int NU = 4;
constexpr int NH = 10;              // Number of control intervals.
constexpr int N = NH + 1;           // Number of state columns in TinyMPC.
constexpr tinytype U0 = tinytype(10.5916);

inline tinyMatrix dynamics_A() {
    static const double rows[NX][NX] = {
        {1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.},
        {0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.},
        {0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.},
        {0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.},
        {0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.},
        {0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992},
        {0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.},
        {0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.},
        {0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.},
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846}
    };
    tinyMatrix A(NX, NX);
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) A(i,j) = tinytype(rows[i][j]);
    return A;
}

inline tinyMatrix dynamics_B() {
    static const double rows[NX][NU] = {
        {0., -0.0726, 0., 0.0726},
        {-0.0726, 0., 0.0726, 0.},
        {-0.0152, 0.0152, -0.0152, 0.0152},
        {0., -0.0006, 0., 0.0006},
        {0.0006, 0., -0.0006, 0.},
        {0.0106, 0.0106, 0.0106, 0.0106},
        {0., -1.4512, 0., 1.4512},
        {-1.4512, 0., 1.4512, 0.},
        {-0.3049, 0.3049, -0.3049, 0.3049},
        {0., -0.0236, 0., 0.0236},
        {0.0236, 0., -0.0236, 0.},
        {0.2107, 0.2107, 0.2107, 0.2107}
    };
    tinyMatrix B(NX, NU);
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NU; ++j) B(i,j) = tinytype(rows[i][j]);
    return B;
}

inline tinyVector q_diagonal() {
    tinyVector q(NX);
    q << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
    return q;
}

inline tinyVector r_diagonal() {
    return tinyVector::Constant(NU, tinytype(0.1));
}

inline tinyVector reference_state() {
    tinyVector ref = tinyVector::Zero(NX);
    ref(2) = tinytype(1.0);
    return ref;
}

// The original OSQP example uses +/- infinity for free coordinates.
// Finite surrogate bounds avoid inf arithmetic in embedded-oriented projections.
// These are effectively unbounded for this hover maneuver, but validate by
// checking that the computed state is many orders of magnitude below 1e9.
constexpr tinytype FREE_BOUND = tinytype(1e9);
inline void state_bounds(tinyVector& lower, tinyVector& upper) {
    lower = tinyVector::Constant(NX, -FREE_BOUND);
    upper = tinyVector::Constant(NX,  FREE_BOUND);
    const tinytype tilt_limit = tinytype(3.14159265358979323846 / 6.0);
    lower(0) = -tilt_limit; upper(0) = tilt_limit;
    lower(1) = -tilt_limit; upper(1) = tilt_limit;
    lower(5) = tinytype(-1.0);
}

inline void input_bounds(tinyVector& lower, tinyVector& upper) {
    lower = tinyVector::Constant(NU, tinytype(9.6) - U0);
    upper = tinyVector::Constant(NU, tinytype(13.0) - U0);
}

}  // namespace quadcopter
