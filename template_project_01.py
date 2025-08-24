# Starter Code for Simplified PINN Project
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 1. Numerical solver for 1D heat equation using implicit scheme
def solve_heat_equation_fd(nx=20, nt=100, L=1.0, T=1.0, alpha=1.0):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(0, L, nx)
    u = np.zeros((nt, nx))
    u[0, :] = np.sin(np.pi * x)  # initial condition

    r = alpha * dt / dx**2
    A = np.eye(nx) * (1 + 2 * r)
    for i in range(1, nx - 1):
        A[i, i - 1] = -r
        A[i, i + 1] = -r

    for n in range(0, nt - 1):
        u[n + 1, :] = np.linalg.solve(A, u[n, :])

    return x, np.linspace(0, T, nt), u


# 2. Supervised NN to learn u(x,t) from FD data
def create_data_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    return model


# 3. PINN model: Custom training loop using tf.GradientTape
def create_pinn_model():
    model =   # ToDo: create the PINN model

    return model


def pinn_loss(model, x_in, t_in, alpha=1.0):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_in, t_in])
        xt = tf.stack([x_in[:, 0], t_in[:, 0]], axis=1)
        u = model(xt)
        u_x = tape2.gradient(u, x_in)
        u_t =   # ToDo: compute derivative for t
        u_xx =   # ToDo: compute 2nd derivate for x
        f = u_t - alpha * u_xx

    return tf.reduce_mean(tf.square(f))


# Example usage
def main():
    x, t, u = solve_heat_equation_fd()
    X, T = np.meshgrid(x, t)
    XT = np.vstack([X.flatten(), T.flatten()]).T
    U = u.flatten()[:, None]

    model = create_data_model()
    model.fit(XT, U, epochs=200, verbose=1)


if __name__ == "__main__":
    main()
