"""
iLQR for Bicycle Model (augmented)
Implements augmented discrete bicycle model [x, y, theta, delta] with inputs [v, dot_delta].
Uses ilqr module for cost computation, backward pass, and forward pass.
"""

import numpy as np
import matplotlib.pyplot as plt

from ilqr import (
    compute_cost,
    compute_cost_coefficients,
    backward_pass_tv,
    forward_pass_nonlinear,
)

# Parameters from ilqr_bicycle_model.md
L = 2.8
DELTA_MAX = 0.5
DELTA_MIN = -0.5
DOT_DELTA_MAX = 0.25

# Default cost matrices for augmented model [x, y, theta, delta], [v, dot_delta]
# Q: state penalty (tracking error), R: control penalty (velocity, dot_delta)
# Tuned so max velocity ~1.0 and dot_delta < DOT_DELTA_MAX (see step 6)
Q_COST = np.diag([1.0, 1.0, 0.0, 0.0])  # [x, y, theta, delta]
R_COST = np.diag([0.1, 1.0])  # [v, dot_delta] - strong penalty on steering rate


def bicycle_dynamics_augmented(x, u, dt):
    """
    Augmented bicycle model: state includes delta so we can penalize dot_delta.
    State: x = [x, y, theta, delta]
    Input: u = [v, dot_delta]
    """
    x_pos, y_pos, theta, delta = x[0], x[1], x[2], x[3]
    v, dot_delta = u[0], u[1]

    x_next = x_pos + dt * v * np.cos(theta)
    y_next = y_pos + dt * v * np.sin(theta)
    theta_next = theta + dt * v * np.tan(delta) / L
    delta_next = delta + dt * dot_delta

    return np.array([x_next, y_next, theta_next, delta_next])


def bicycle_diff_dynamics_augmented(x, u, dt):
    """
    Jacobians for augmented model. Returns A (4x4), B (4x2).
    """
    theta, delta = x[2], x[3]
    v, dot_delta = u[0], u[1]

    # A = ∂f/∂x (4x4)
    A = np.array(
        [
            [1, 0, -dt * v * np.sin(theta), 0],
            [0, 1, dt * v * np.cos(theta), 0],
            [0, 0, 1, dt * v / (L * np.cos(delta) ** 2)],
            [0, 0, 0, 1],
        ]
    )

    # B = ∂f/∂u (4x2)
    B = np.array(
        [
            [dt * np.cos(theta), 0],
            [dt * np.sin(theta), 0],
            [dt * np.tan(delta) / L, 0],
            [0, dt],
        ]
    )

    return A, B


def simulate_trajectory(x0, us, dt):
    """Simulate trajectory given initial state and control sequence (augmented model)."""
    N = len(us)
    xs = np.zeros((N + 1, 4))
    xs[0] = x0

    for t in range(N):
        xs[t + 1] = bicycle_dynamics_augmented(xs[t], us[t], dt)

    return xs


def generate_circular_trajectory(radius, velocity, dt, num_steps):
    """
    Generate constant v, dot_delta=0 inputs for circular trajectory (augmented model).
    For circle: R = L / tan(delta) => delta = arctan(L/R)
    """
    delta = np.arctan(L / radius)
    delta = np.clip(delta, DELTA_MIN, DELTA_MAX)

    us = np.zeros((num_steps, 2))
    us[:, 0] = velocity
    us[:, 1] = 0.0  # dot_delta
    return us


def generate_figure8_reference(A, B, v_ref, dt, num_steps):
    """
    Generate infinite-symbol (figure-8) reference trajectory.
    Parametric: x(s) = A*sin(s), y(s) = B*sin(2*s), s in [0, 2π].
    Returns xs_ref (N+1, 4), us_ref (N, 2) for augmented model.

    Reference is sampled at uniform time steps t = 0, dt, 2*dt, ..., N*dt so that
    index i corresponds to simulation time i*dt. v_ref is set to path_length/(N*dt)
    so the reference completes one loop in the simulation horizon (v_ref arg ignored).
    """
    # Fine s-grid for arc-length integration and interpolation
    n_fine = 2000
    s_fine = np.linspace(0, 2 * np.pi, n_fine + 1, endpoint=True)

    # Position and derivatives on fine grid
    x_fine = A * np.sin(s_fine)
    y_fine = B * np.sin(2 * s_fine)
    dx_ds = A * np.cos(s_fine)
    dy_ds = 2 * B * np.cos(2 * s_fine)

    path_speed_sq = dx_ds**2 + dy_ds**2
    path_speed_sq = np.maximum(path_speed_sq, 1e-8)
    path_speed = np.sqrt(path_speed_sq)
    path_speed = np.maximum(path_speed, 1e-6)

    # Cumulative arc length: u(s) = integral_0^s |dr/ds| ds
    arc_length = np.concatenate([[0], np.cumsum(path_speed[:-1] * np.diff(s_fine))])
    L_path = arc_length[-1]

    # v_ref so that we complete one loop in t_sim = num_steps * dt
    v_ref = L_path / (num_steps * dt)

    # Sample at uniform time: t_i = i*dt => arc_length(s_i) = v_ref * t_i
    t_vals = np.arange(num_steps + 1) * dt
    arc_at_t = v_ref * t_vals
    arc_at_t = np.minimum(arc_at_t, L_path - 1e-10)  # avoid extrapolation at endpoint

    # Map arc_length -> s via interpolation
    s_vals = np.interp(arc_at_t, arc_length, s_fine)

    # Evaluate reference at s_vals
    x_ref = A * np.sin(s_vals)
    y_ref = B * np.sin(2 * s_vals)
    dx_ds_at_s = A * np.cos(s_vals)
    dy_ds_at_s = 2 * B * np.cos(2 * s_vals)

    theta_ref = np.arctan2(dy_ds_at_s, dx_ds_at_s)
    for i in range(1, len(theta_ref)):
        d = theta_ref[i] - theta_ref[i - 1]
        if d > np.pi:
            theta_ref[i:] -= 2 * np.pi
        elif d < -np.pi:
            theta_ref[i:] += 2 * np.pi

    d2x_ds2 = -A * np.sin(s_vals)
    d2y_ds2 = -4 * B * np.sin(2 * s_vals)
    path_speed_sq_at_s = dx_ds_at_s**2 + dy_ds_at_s**2
    path_speed_sq_at_s = np.maximum(path_speed_sq_at_s, 1e-8)
    kappa = (dx_ds_at_s * d2y_ds2 - dy_ds_at_s * d2x_ds2) / (path_speed_sq_at_s**1.5)

    delta_ref = np.arctan(L * kappa)
    delta_ref = np.clip(delta_ref, DELTA_MIN, DELTA_MAX)

    path_speed_at_s = np.sqrt(path_speed_sq_at_s)
    path_speed_at_s = np.maximum(path_speed_at_s, 1e-6)
    ds_dt = v_ref / path_speed_at_s

    xs_ref = np.column_stack([x_ref, y_ref, theta_ref, delta_ref])

    # dot_delta = d(delta)/dt = d(delta)/ds * ds/dt
    dot_delta_ref = np.gradient(delta_ref, s_vals) * ds_dt
    dot_delta_ref = np.clip(dot_delta_ref, -DOT_DELTA_MAX, DOT_DELTA_MAX)

    us_ref = np.zeros((num_steps, 2))
    us_ref[:, 0] = v_ref
    us_ref[:, 1] = dot_delta_ref[:-1]

    return xs_ref, us_ref


def _ilqr_backward_pass(xs, us, xs_ref, us_ref, dt, Q, R, Ks, ds):
    """iLQR backward pass for bicycle model via ilqr.backward_pass_tv."""
    def get_AB(i):
        return bicycle_diff_dynamics_augmented(xs[i], us[i], dt)

    def get_l_coeffs(i):
        return compute_cost_coefficients(
            xs[i], us[i], xs_ref[i], us_ref[i], Q, R
        )

    backward_pass_tv(
        get_AB, get_l_coeffs, xs, us, xs_ref, us_ref, Q, R, Ks, ds
    )


def _ilqr_forward_pass(x0, xs, us, Ks, ds, dt, alpha=1.0):
    """iLQR forward pass for bicycle model via ilqr.forward_pass_nonlinear."""
    forward_pass_nonlinear(
        x0, bicycle_dynamics_augmented, dt, Ks, ds, xs, us, alpha
    )


def plot_circular_trajectory(x0, xs, us, radius, dt, save_path="circular_trajectory.png", show=False):
    """Plot circular trajectory and control inputs (augmented model)."""
    N = len(us)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xs[:, 0], xs[:, 1], "b-", linewidth=2, label="Trajectory")
    axes[0].plot(x0[0], x0[1], "go", markersize=10, label="Start")
    axes[0].plot(xs[-1, 0], xs[-1, 1], "ro", markersize=10, label="End")
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    axes[0].plot(
        radius * np.cos(theta_circle),
        radius * np.sin(theta_circle),
        "k--",
        alpha=0.5,
        label="Reference circle",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Circular Trajectory (Augmented Bicycle Model)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True)

    ts = np.arange(N + 1) * dt
    axes[1].plot(ts[:-1], us[:, 0], "b-", label="v")
    axes[1].plot(ts[:-1], us[:, 1], "r-", label="dot_delta")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Input")
    axes[1].set_title("Control Inputs")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_figure8_tracking(xs_ref, xs_nom, us_ref, us_nom, x0_aug, dt, save_path="figure8_ilqr.png"):
    """Plot figure-8 tracking results."""
    N = len(us_nom)
    ts = np.arange(N + 1) * dt

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    axes[0, 0].plot(xs_ref[:, 0], xs_ref[:, 1], "k--", alpha=0.7, label="Reference")
    axes[0, 0].plot(xs_nom[:, 0], xs_nom[:, 1], "b-", linewidth=2, label="iLQR")
    axes[0, 0].plot(x0_aug[0], x0_aug[1], "go", markersize=10, label="Start")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_title("Figure-8 Trajectory Tracking")
    axes[0, 0].axis("equal")
    margin = 1.0
    axes[0, 0].set_xlim(xs_ref[:, 0].min() - margin, xs_ref[:, 0].max() + margin)
    axes[0, 0].set_ylim(xs_ref[:, 1].min() - margin, xs_ref[:, 1].max() + margin)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(ts[:-1], us_ref[:, 0], "k--", alpha=0.7, label="v_ref")
    axes[0, 1].plot(ts[:-1], us_ref[:, 1], "g--", alpha=0.7, label="dot_delta_ref")
    axes[0, 1].plot(ts[:-1], us_nom[:, 0], "b-", label="v")
    axes[0, 1].plot(ts[:-1], us_nom[:, 1], "r-", label="dot_delta")
    axes[0, 1].axhline(DOT_DELTA_MAX, color="r", linestyle=":", alpha=0.5)
    axes[0, 1].axhline(-DOT_DELTA_MAX, color="r", linestyle=":", alpha=0.5)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Control")
    axes[0, 1].set_title("Control Inputs")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(ts, xs_nom[:, 0], "b-", label="x")
    axes[1, 0].plot(ts, xs_nom[:, 1], "r-", label="y")
    axes[1, 0].plot(ts, xs_ref[:, 0], "k--", alpha=0.5)
    axes[1, 0].plot(ts, xs_ref[:, 1], "k--", alpha=0.5)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Position")
    axes[1, 0].set_title("State vs Reference")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(ts, xs_ref[:, 3], "g--", alpha=0.7, label="delta_ref")
    axes[1, 1].plot(ts, xs_nom[:, 3], "m-", label="delta")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Delta (rad)")
    axes[1, 1].set_title("Delta (steering angle)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    err = np.linalg.norm(xs_nom[:, :2] - xs_ref[:, :2], axis=1)
    axes[2, 0].plot(ts, err, "b-")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Position error")
    axes[2, 0].set_title("Tracking Error")
    axes[2, 0].grid(True)

    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def run_ilqr_figure8(
    xs_ref, us_ref, x0_aug, dt, Q, R, max_iter=50, epsilon=1e-2, verbose=True
):
    """
    Run iLQR to track figure-8 reference. Returns (xs_nom, us_nom).
    """
    N = len(us_ref)
    xs_nom = xs_ref.copy()
    us_nom = us_ref.copy()
    Ks = np.zeros((N, 2, 4))
    ds = np.zeros((N, 2, 1))

    def total_cost(xs, us):
        J = 0.0
        for i in range(N):
            J += compute_cost(xs[i], us[i], xs_ref[i], us_ref[i], Q, R)
        J += 0.5 * (
            (xs[N] - xs_ref[N]).T @ Q @ (xs[N] - xs_ref[N])
        ).item()
        return J

    J_prev = None
    for j in range(max_iter):
        xs_prev = xs_nom.copy()
        us_prev = us_nom.copy()

        _ilqr_backward_pass(xs_nom, us_nom, xs_ref, us_ref, dt, Q, R, Ks, ds)

        xs_nom[:] = xs_prev
        us_nom[:] = us_prev
        _ilqr_forward_pass(x0_aug, xs_nom, us_nom, Ks, ds, dt, alpha=1.0)
        J = total_cost(xs_nom, us_nom)

        if verbose:
            print(f"  iter {j}: J={J:.4f}")

        if J_prev is not None and abs((J - J_prev) / (J_prev + 1e-10)) < epsilon:
            if verbose:
                print("  Converged.")
            break
        J_prev = J

    return xs_nom, us_nom


def _test_cost_penalty():
    """Verify cost and its derivatives."""
    x = np.array([1.0, 2.0, 0.5, 0.2])
    u = np.array([1.0, 0.1])
    x_ref = np.array([0.0, 0.0, 0.0, 0.0])
    u_ref = np.array([1.0, 0.0])  # penalize dot_delta

    cost = compute_cost(x, u, x_ref, u_ref, Q_COST, R_COST)
    l_xx, l_xu, l_ux, l_uu, l_x, l_u = compute_cost_coefficients(
        x, u, x_ref, u_ref, Q_COST, R_COST
    )

    # Numerical gradient check (tolerance relaxed: forward diff has O(eps) truncation error)
    eps = 1e-6
    tol = 1e-4  # R_COST can be large, so truncation error ~0.5*R_ii*eps
    for i in range(4):
        x_plus = x.copy()
        x_plus[i] += eps
        cost_plus = compute_cost(x_plus, u, x_ref, u_ref, Q_COST, R_COST)
        l_x_num = (cost_plus - cost) / eps
        assert abs(l_x[i] - l_x_num) < tol, f"l_x[{i}] mismatch"
    for i in range(2):
        u_plus = u.copy()
        u_plus[i] += eps
        cost_plus = compute_cost(x, u_plus, x_ref, u_ref, Q_COST, R_COST)
        l_u_num = (cost_plus - cost) / eps
        assert abs(l_u[i] - l_u_num) < tol, f"l_u[{i}] mismatch"

    print("Cost penalty check passed.")


if __name__ == "__main__":
    _test_cost_penalty()

    # --- Circular trajectory (augmented model verification) ---
    dt = 0.1
    t_sim = 10.0
    N = int(t_sim / dt)
    radius = 6.0
    velocity = 1.0

    us = generate_circular_trajectory(radius, velocity, dt, N)
    delta_circle = np.arctan(L / radius)
    x0_aug = np.array([radius, 0.0, np.pi / 2, delta_circle])
    xs = simulate_trajectory(x0_aug, us, dt)

    plot_circular_trajectory(x0_aug, xs, us, radius, dt)
    print(f"Trajectory: {N} steps, dt={dt}s")
    print(f"Radius={radius}, v={velocity}, delta={delta_circle:.4f} rad")
    print(f"Start: {x0_aug[:3]}, End: {xs[-1, :3]}")

    # --- iLQR figure-8 tracking ---
    print("\n--- iLQR Figure-8 Tracking ---")
    dt_ilqr = 0.1
    t_sim_ilqr = 20.0
    N_ilqr = int(t_sim_ilqr / dt_ilqr)
    A_fig8, B_fig8 = 20.0, 10.0

    xs_ref, us_ref = generate_figure8_reference(
        A_fig8, B_fig8, v_ref=1.0, dt=dt_ilqr, num_steps=N_ilqr
    )
    x0_aug = xs_ref[0].copy()
    x0_aug[0] += 0.3

    xs_nom, us_nom = run_ilqr_figure8(
        xs_ref, us_ref, x0_aug, dt_ilqr, Q_COST, R_COST
    )

    plot_figure8_tracking(xs_ref, xs_nom, us_ref, us_nom, x0_aug, dt_ilqr)
    print(f"Max velocity: {np.max(us_nom[:, 0]):.3f}")
    print(f"Max |dot_delta|: {np.max(np.abs(us_nom[:, 1])):.3f}")
