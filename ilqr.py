import numpy as np

# Initial Conditions
x0 = np.array([[3], [0]])

# System Dynamics (Continuous)
A_continue = np.array([[0, 1], [0.01, 0]])
B_continue = np.array([[0], [1]])

# Control Law
Q = np.array([[1.0, 0], [0, 0.5]]) # Penalize angular error & angular velocity
R = np.array([0.1]) # Penalize thrust effort

# Descretization Parameter
t_sim = 30
N = 200
ts = np.linspace(0, t_sim, N)

# descritize A_continues and B_continue
delta_t = float(t_sim / N)
A = np.eye(2) + A_continue * delta_t
B = B_continue * delta_t
print(f"{A=}")
print(f"{B=}")

if __name__ == "__main__":
    # xs is N multiples of x0
    xs = np.zeros((N, x0.shape[0]))
    xs[0] = x0.T
    # xs = np.tile(x0.T, (N, 1))
    Ks = np.zeros((N, 2)) # feedback gain K_t, from backward pass
    ds = np.zeros((N, 1)) # feedforward term d_t, from backward pass
    us = np.zeros((N, 1))
    
    # Implement Here by Agent

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ts, xs[:, 0])
    ax[0].set_ylim([-0.5, 3.5])

    accel = np.diff(xs[:, 1]/delta_t)
    ax[1].plot(ts[:-1], accel)
    ax[1].set_ylim([-10, 5])

    fuel_approx = np.sum(abs(accel)*1.0) * delta_t # Under the assumption that inertial is 1.0
    print('Total fuel_approx used:', fuel_approx)
    # fuel = torque*angle = acceleration*inertia*angle
    fuel = np.sum(abs(accel)*1.0*xs[:-1, 0]) * delta_t
    print('Fuel used:', fuel)
    # save the fig to .png file
    plt.savefig('result.png')