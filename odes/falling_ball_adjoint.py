"""
This module provides the parametrized ODEs of the falling object
problem and its solution using the adjoint method.

The goal is to estimate the unknowns using gradient descent. The cost
function is defined as the squared error between the true height of the
object and the predicted height. The optimization is formed as a
constrained optimization problem where the constraint is the ODE of the
falling object (ode_f).

To find the gradient of the cost function, the adjoint method is used.
See the following references for more information:
- https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/ocw_18s096_lecture06-part1_2023jan30_mp4/
- https://www.youtube.com/watch?v=k6s2G5MZv-I&ab_channel=MachineLearning%26Simulation
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# noinspection PyUnusedLocal
def ode_f(t, u_t, p) -> np.ndarray:
    """
    The ODE function of a falling object:
        du1/dt = f1(u, t; p) = u2,
        du2/dt = f2(u, t; p) = p1,
    where t is time, u is the vector of the height and velocity of the
    object [x, v], and p is the vector of the gravitational acc. and
    the initial velocity [g, v0].

    Args:
        t: current time
        u_t: array of height and velocity of the object evaluated at t
        p: parameter array containing the gravitational acceleration
            and initial velocity
    Returns:
        du/dt evaluation at t
    """
    return np.array([u_t[1], p[0]])


def ode_lambda(t, lambda_t, u_sol, u_true) -> np.ndarray:
    """
    The lambda ODE:
        d_lambda(t)/dt^T = -lambda(t)^T * (del_f/del_u) - del_h/del_u

    Args:
        t: current time
        lambda_t: the value of lambda at time t
        u_sol: 1-D interpolation of the solution u at all t
        u_true: 1-D interpolation of the true value of u at all t
    Returns:
        d_lambda/dt evaluated at t
    """
    del_f_del_u = np.array([[0, 1], [0, 0]])
    del_h_del_u = np.array([1, 0]) * (u_sol(t) - u_true(t))
    # Numpy handles the transpose of the lambda vector automatically.
    return -lambda_t @ del_f_del_u - del_h_del_u


# noinspection PyUnusedLocal
def dw(t, w, lamda) -> np.ndarray:
    """
    The integration term of the cost function gradient.
        dw/dt = del_h/del_p + lambda^T * del_f/del_p

    Args:
        t: current time
        w: the value of w at time t
        lamda: 1-D interpolation of the solution lambda at all t
    """
    # del_h_del_p = np.array([0, 0])
    del_f_del_p = np.array([[0, 0],[1, 0]])
    return lamda(t) @ del_f_del_p


# noinspection PyUnresolvedReferences,PyPep8Naming
def optimize(v0, g0, ut, t_eval, eta, num_epochs):
    """
    Solve the ODEs for the falling object.

    Args:
        v0: initial guess of the velocity
        g0: initial guess of the gravitational acceleration
        ut: true height of the object at all t
        t_eval: time stamps
        eta: learning rate
        num_epochs: number of iterations

    Returns:
        The optimized parameters, predicted solutions, and costs.
    """
    zero = np.array([0, 0])
    p = np.array([g0, v0])

    costs = []
    pred_gv = []
    sols = []

    interp_opts = dict(
        fill_value='extrapolate', copy=False, assume_sorted=True)

    for num_epochs in range(num_epochs):
        pred_gv.append([p[0], p[1]])

        u0 = np.array([ut[0], p[1]])
        sol_u = solve_ivp(
            ode_f, t_span=(t_eval[0], t_eval[-1]), y0=u0, args=(p,),
            t_eval=t_eval)

        # Solve the terminal value problem for lambda.
        fun_u = interp1d(t_eval, sol_u.y[0], **interp_opts)
        fun_u_true = interp1d(t_eval, ut, **interp_opts)
        # The solution is backward in time since t_eval is decreasing.
        sol_lam = solve_ivp(
            ode_lambda, t_span=(t_eval[-1], t_eval[0]), y0=zero,
            args=(fun_u, fun_u_true), t_eval=t_eval[::-1])

        # Sort the lambda solution forward in time since the ODE takes
        # care of the time stamps in increasing order.
        lam = sol_lam.y[:,::-1]
        fun_lambda = interp1d(t_eval, lam, **interp_opts)
        sol_w = solve_ivp(
            dw, t_span=(t_eval[0], t_eval[-1]), y0=zero,
            args=(fun_lambda,), t_eval=t_eval)

        dJ_dp = sol_w.y[:, -1] + lam[:, 0] @ np.array([[0, 0], [0, 1]])
        p -= eta * dJ_dp

        costs.append(0.5 * np.sum((ut - sol_u.y[0]) ** 2))
        sols.append(sol_u.y[0])

    return np.array(pred_gv), sols, np.array(costs)
