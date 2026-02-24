"""
iLQR (iterative Linear Quadratic Regulator) - Core library.
Provides cost computation, backward pass, and forward pass for both
linear time-invariant and nonlinear/time-varying systems.
"""

import numpy as np


def compute_cost(x, u, x_ref, u_ref, Q, R):
    """
    Running cost: 0.5*(x-x_ref)^T Q (x-x_ref) + 0.5*(u-u_ref)^T R (u-u_ref)
    For regulation (no reference): use x_ref=0, u_ref=0.
    """
    dx = (x - x_ref).reshape(-1, 1)
    du = (u - u_ref).reshape(-1, 1)
    return 0.5 * (dx.T @ Q @ dx + du.T @ R @ du).item()


def compute_cost_coefficients(x, u, x_ref, u_ref, Q, R):
    """
    Cost derivatives for iLQR backward pass.
    l(x,u) = 0.5*(x-x_ref)^T Q (x-x_ref) + 0.5*(u-u_ref)^T R (u-u_ref)
    Returns l_xx, l_xu, l_ux, l_uu, l_x, l_u.
    For regulation: use x_ref=0, u_ref=0.
    """
    dx = (x - x_ref).reshape(-1, 1)
    du = (u - u_ref).reshape(-1, 1)

    l_xx = Q.copy()
    l_uu = np.atleast_2d(R).copy()
    l_xu = np.zeros((Q.shape[0], l_uu.shape[1]))
    l_ux = l_xu.T

    l_x = (Q @ dx).flatten()
    l_u = (np.atleast_2d(R) @ du).flatten()

    return l_xx, l_xu, l_ux, l_uu, l_x, l_u


def backward_pass_linear(A, B, Q, R, xs, us, Ks, ds, x_ref=None, u_ref=None):
    """
    iLQR backward pass for linear time-invariant system: x_{t+1} = A x_t + B u_t.
    K_t = -Q_uu^{-1} Q_ux, d_t = -Q_uu^{-1} Q_u
    xs: (N, n_state), us: (N, n_ctrl). If x_ref, u_ref are None, use zeros (regulation).
    """
    N = len(us)
    if x_ref is None:
        x_ref = np.zeros_like(xs)
    if u_ref is None:
        u_ref = np.zeros_like(us)

    x_terminal = xs[N - 1] - x_ref[N - 1]
    p = (Q @ x_terminal).reshape(-1, 1)
    P = Q.copy()

    for i in range(N - 2, -1, -1):
        l_xx, l_xu, l_ux, l_uu, l_x, l_u = compute_cost_coefficients(
            xs[i], us[i], x_ref[i], u_ref[i], Q, R
        )

        Q_xx = l_xx + A.T @ P @ A
        Q_xu = l_xu + A.T @ P @ B
        Q_ux = l_ux + B.T @ P @ A
        Q_uu = l_uu + B.T @ P @ B
        Q_x = (l_x.reshape(-1, 1) + A.T @ p).flatten()
        Q_u = (l_u.reshape(-1, 1) + B.T @ p).flatten()

        Q_uu_inv = np.linalg.inv(Q_uu)
        K = -Q_uu_inv @ Q_ux
        d = (-Q_uu_inv @ Q_u.reshape(-1, 1)).reshape(-1, 1)

        Ks[i] = K.flatten()
        ds[i] = d

        P = Q_xx + Q_xu @ K + K.T @ Q_ux + K.T @ Q_uu @ K
        p = (
            Q_x.reshape(-1, 1)
            + (Q_u.reshape(1, -1) @ K).T
            + Q_xu @ d
            + K.T @ (Q_uu @ d)
        )


def backward_pass_tv(get_AB, get_l_coeffs, xs, us, xs_ref, us_ref, Q, R, Ks, ds):
    """
    iLQR backward pass for time-varying/nonlinear system.
    get_AB(t) -> (A_t, B_t), get_l_coeffs(t) -> (l_xx, l_xu, l_ux, l_uu, l_x, l_u)
    xs: (N+1, n_state), us: (N, n_ctrl)
    """
    N = len(us)

    x_terminal = xs[N] - xs_ref[N]
    p = (Q @ x_terminal).reshape(-1, 1)
    P = Q.copy()

    for i in range(N - 1, -1, -1):
        A, B = get_AB(i)
        l_xx, l_xu, l_ux, l_uu, l_x, l_u = get_l_coeffs(i)

        Q_xx = l_xx + A.T @ P @ A
        Q_xu = l_xu + A.T @ P @ B
        Q_ux = l_ux + B.T @ P @ A
        Q_uu = l_uu + B.T @ P @ B
        Q_x = (l_x.reshape(-1, 1) + A.T @ p).flatten()
        Q_u = (l_u.reshape(-1, 1) + B.T @ p).flatten()

        Q_uu_inv = np.linalg.inv(Q_uu)
        K = -Q_uu_inv @ Q_ux
        d = (-Q_uu_inv @ Q_u.reshape(-1, 1)).reshape(-1, 1)

        Ks[i] = K
        ds[i] = d

        P = Q_xx + Q_xu @ K + K.T @ Q_ux + K.T @ Q_uu @ K
        p = (
            Q_x.reshape(-1, 1)
            + (Q_u.reshape(1, -1) @ K).T
            + Q_xu @ d
            + K.T @ (Q_uu @ d)
        )


def forward_pass_linear(x0, A, B, Ks, ds, xs, us):
    """
    iLQR forward pass for linear system: x_{t+1} = A x_t + B u_t.
    delta_u_t = K_t delta_x_t + d_t, u_t = u_prev + delta_u_t
    """
    xs_prev = xs.copy()
    us_prev = us.copy()
    xs[0] = x0.flatten() if x0.ndim > 1 else x0

    for t in range(len(us) - 1):
        delta_x_t = xs[t] - xs_prev[t]
        delta_u_t = (Ks[t] @ delta_x_t).reshape(-1, 1) + ds[t]
        us[t] = us_prev[t] + delta_u_t.flatten()
        xs[t + 1] = (A @ xs[t].reshape(-1, 1) + B @ us[t].reshape(-1, 1)).flatten()


def forward_pass_nonlinear(x0, dynamics_fn, dt, Ks, ds, xs, us, alpha=1.0):
    """
    iLQR forward pass for nonlinear system: x_{t+1} = dynamics_fn(x_t, u_t, dt).
    delta_u_t = alpha * (K_t delta_x_t + d_t), u_t = u_prev + delta_u_t
    """
    N = len(us)
    xs_new = np.zeros_like(xs)
    us_new = np.zeros_like(us)
    xs_new[0] = x0

    for t in range(N):
        delta_x = xs_new[t] - xs[t]
        delta_u = alpha * (Ks[t] @ delta_x.reshape(-1, 1) + ds[t])
        delta_u = delta_u.flatten()

        us_new[t] = us[t] + delta_u
        xs_new[t + 1] = dynamics_fn(xs_new[t], us_new[t], dt)

    xs[:] = xs_new
    us[:] = us_new


# --- Original ilqr demo (linear system) ---
if __name__ == "__main__":
    """
    This is a simple example of UFO rotation control. Please refer to https://www.youtube.com/watch?v=E_RDCFOlJx4 (in English)
    or https://www.bilibili.com/video/BV1bF41197HJ/?spm_id_from=333.1391.0.0&vd_source=5d69ac08bd10beeee8c1070d4d354bed (in Chinese)
    """
    # Initial Conditions
    x0 = np.array([[3], [0]])

    # System Dynamics (Continuous)
    A_continue = np.array([[0, 1], [0.01, 0]])
    B_continue = np.array([[0], [1]])

    # Control Law
    Q = np.array([[1.0, 0], [0, 0.5]])
    R = np.array([[0.1]])

    # Discretization
    t_sim = 30
    N = 200
    ts = np.linspace(0, t_sim, N)
    delta_t = float(t_sim / N)
    A = np.eye(2) + A_continue * delta_t
    B = B_continue * delta_t
    print(f"{A=}")
    print(f"{B=}")

    xs = np.zeros((N, x0.shape[0]))
    xs[0] = x0.T.flatten()
    Ks = np.zeros((N, 2))
    ds = np.zeros((N, 1))
    us = np.zeros((N, 1))

    # Initialize nominal trajectory
    for i in range(N - 1):
        xs[i + 1] = (A @ xs[i].reshape(-1, 1) + B @ us[i].reshape(-1, 1)).flatten()

    xs_ref = np.zeros_like(xs)
    u_ref = np.zeros_like(us)

    max_iter = 100
    epsilon = 1e-6
    J_prev = None

    for j in range(max_iter):
        backward_pass_linear(A, B, Q, R, xs, us, Ks, ds, xs_ref, u_ref)
        forward_pass_linear(x0.flatten(), A, B, Ks, ds, xs, us)

        J = 0.0
        for i in range(N):
            x = xs[i].reshape(-1, 1)
            u = us[i].reshape(-1, 1)
            J += (x.T @ Q @ x + u.T @ R @ u).item()

        print(f"iter {j}: J={J:.6f}")

        if J_prev is not None and abs((J - J_prev) / J_prev) < epsilon:
            break
        J_prev = J

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ts, xs[:, 0])
    ax[0].set_ylim([-0.5, 3.5])

    accel = np.diff(xs[:, 1] / delta_t)
    ax[1].plot(ts[:-1], accel)
    ax[1].set_ylim([-10, 5])

    fuel_approx = np.sum(abs(accel) * 1.0) * delta_t
    print("Total fuel_approx used:", fuel_approx)
    fuel = np.sum(abs(accel) * 1.0 * xs[:-1, 0]) * delta_t
    print("Fuel used:", fuel)
    plt.savefig("result.png")
