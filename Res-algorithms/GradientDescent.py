import matplotlib.pyplot as plt
import numpy as np
import random

def GradientDescent(lhs, rhs, initial_guess, max_iter=100, epsilon=0.01):
    """
    gradient descent optimization algorithm
    Args:
        lhs: numpy.ndarray - left-hand side matrix
        rhs: np.ndarray - right-hand side matrix
        initial_guess: np.ndarray - initial guess for the solution
        max_iter:
        epsilon:

    Returns: np.ndarray - optimal solution
    """
    x = initial_guess
    iter = 0
    res = lhs.T @ (rhs - lhs @ x)
    mse = res.T @ res
    mse0 = mse
    min_mse = mse0
    min_x = initial_guess

    while iter < max_iter and mse > epsilon ** 2 * mse0:
        res = lhs.T @ (rhs - lhs @ x)
        prev_x = x
        prev_mse = mse
        x = x + res
        mse = res.T @ res
        #if mse > prev_mse:
            #x = prev_x
            #break
        if iter % 25 == 0:
            print(f"Gradient Descent iteration {iter} mean-squared error: {mse:.4f}")
        iter += 1
    return x

def conjugate_gradient(A, b, x0, max_iter=100, tol=1e-6):
    x = x0
    r = b - np.dot(A, x)
    p = r.copy()
    for _ in range(max_iter):
        alpha = np.dot(r, r) / np.dot(p, np.dot(A, p))
        x += alpha * p
        r_new = r - alpha * np.dot(A, p)
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x


def kaczmarz(A, b, max_iter=1000, tol=1e-6, x0=None):
    """ Kaczmarz algorithm for solving Ax = b """
    logx = []
    logdiff = []
    m, n = A.shape
    x = x0
    if x0 is None:
        x = np.zeros(n)  # Initial guess

    for k in range(max_iter):
        i = k % m  # Cycle through rows
        a_i = A[i, :]
        cols = a_i.nonzero()[1]
        values = a_i.data
        x_i = x[cols]
        scalar1 = values.T @ x_i
        a_i_norm = np.linalg.norm(values)
        x[cols] += (b[i] - scalar1) / a_i_norm * values

        # Check for convergence
        diff = np.linalg.norm(A @ x - b)
        logx.append(np.log(np.linalg.norm(x)))
        logdiff.append(np.log(diff))
        if diff < tol:
            break
        if k % 125 == 0:
            print("error:", diff)


    plt.plot(logdiff, logx)
    plt.show()

    return x