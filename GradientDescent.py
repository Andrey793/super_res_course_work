
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

    while iter < max_iter and mse > epsilon ** 2 * mse0:
        res = lhs.T @ (rhs - lhs @ x)
        x = x + res
        mse = res.T @ res
        print(f"Gradient Descent iteration {iter} mean-squared error: {mse:.4f}")
        iter += 1
    return x