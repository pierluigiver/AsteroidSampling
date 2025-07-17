import cvxpy as cp
import numpy as np
import mujoco
from scipy.stats import f
import sympy as sp

def compute_gain_lmi_paper_version(lambda_val, Ts):
    n, m = 3, 2  # State and control dimensions

    # Define ranges from the paper
    k_vals = [300, 2.7e6]  # N/m
    c_vals = [0.35, 1000] # Ns/m

    A_list = []
    B_list = []

    for k in k_vals:
        for c in c_vals:

            A = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, -k, 0]
            ])
            A_list.append(A)

            B = np.array([
                [1, 0],
                [0, 1],
                [0, -c]
            ])
            B_list.append(B)


    # Define LMI variables
    Q = cp.Variable((n, n))
    Y = cp.Variable((m, n))
    s = cp.Variable()

    decay = np.exp(-2 * lambda_val * Ts)
    I = np.eye(n)
    Z = np.zeros((n, n))

    constraints = []

    for A in A_list:
        for B in B_list:
            M11 = decay * Q
            M12 = A @ Q + B @ Y
            M21 = M12.T
            M22 = Q
            LHS = cp.bmat([[M11, M12],
                           [M21, M22]])
            RHS = cp.bmat([[s * I, Z],
                           [Z, Z]])
            constraints.append(LHS >> RHS)


    constraints += [Q >> s * I, s >= 0]

    # Solve
    prob = cp.Problem(cp.Minimize(s), constraints)
    
    prob.solve(solver=cp.MOSEK, verbose=True)
    
    K = Y.value @ np.linalg.pinv(Q.value)

    return K


def pcac_forgetting_law(z_buffer, tau_n, tau_d, p, eta, alpha):

    zn = z_buffer[-(tau_n+1):, :]   # (tau_n+1, p)
    zd = z_buffer[-(tau_d+1):, :]   # (tau_d+1, p)
    Sigma_n = (zn.T @ zn) / (tau_n + 1)
    Sigma_d = (zd.T @ zd) / (tau_d + 1)
    LH = (tau_n / tau_d) * np.trace(Sigma_n @ np.linalg.inv(Sigma_d))
    a = (tau_n + tau_d - p - 1) * (tau_d - 1) / ((tau_d - p - 3) * (tau_d - p))
    b = 4 + (p * tau_n + 2) / (a - 1)
    c = p * tau_n * (b - 2) / (b * (tau_d - p - 1))
    F_inv = f.ppf(1 - alpha, p * tau_n, int(b))
    g = np.sqrt(LH / c) - np.sqrt(F_inv)
    beta_k = 1 + eta * g * (g > 0)
    lambda_k = 1 / beta_k

    return lambda_k

def pcac_rls_update_mimo(t, y, tau, P_k, w_k, p, m, n_window, tau_n, tau_d, eta, alpha):
    """
    Standard block RLS update with variable forgetting (for MIMO, Mohseni, corrected denominator and update law)
    Args:
        t       : time step
        y       : output matrix (p, n_window)
        tau     : input matrix (m, n_window)
        P_k     : covariance matrix (p*n_window*(p+m), p*n_window*(p+m))
        w_k     : parameter vector (p*n_window*(p+m), 1)
        p       : number of outputs
        m       : number of inputs
        n_window: window size
        tau_n, tau_d, eta, alpha: forgetting law parameters
    Returns:
        P_k_new, w_k_new
    """

    # Build the regressor row: [-y_{k-1}^T ... -y_{k-n}^T tau_{k-1}^T ... tau_{k-n}^T]
    phi_row = []
    y = y.reshape(p, n_window)  # Ensure y is (p, n_window)
    tau = tau.reshape(m, n_window)  # Ensure tau is (m, n_window)
    for i in range(n_window):
        phi_row.append(-y[:, i].T)      # shape: (p,)
    for i in range(n_window):
        phi_row.append(tau[:, i].T)     # shape: (m,)

    phi_row = np.hstack(phi_row).reshape(1, -1)  # shape: (1, n_window*(p+m))

    # Kronecker product with I_p
    phi_k = np.kron(phi_row, np.eye(p))  # shape: (p, p*n_window*(p+m))

    # Compute performance variable (current output minus predicted output)
    z_k = y - phi_k @ w_k  # (p, 1)
    
    # Compute forgetting factor
    lambda_k = pcac_forgetting_law(z_k, tau_n, tau_d, p, eta, alpha)  

    # RLS update
    # Compute the innovation covariance
    S_k = lambda_k * np.eye(p) + phi_k @ P_k @ phi_k.T  # (p, p)
    
    # Update covariance matrix
    P_k_new = lambda_k**(-1) * P_k - lambda_k**(-1) * P_k @ phi_k.T @ np.linalg.inv(S_k) @ phi_k @ P_k
    
    # Update parameter vector
    w_k_new = w_k + P_k_new @ phi_k.T @ z_k

    return P_k_new, w_k_new

def extract_F_G_from_wk(w_k, p, m, n_window):
    F_blocks = []
    G_blocks = []
    offset = 0
    w_k = np.asarray(w_k).flatten(order='F')  # Ensure it's a 1D array
    for i in range(n_window):
        F_block = w_k[offset:offset + p*p].reshape((p, p), order='F')
        F_blocks.append(F_block)
        offset += p*p
    for i in range(n_window):
        G_block = w_k[offset:offset + p*m].reshape((p, m), order='F')
        G_blocks.append(G_block)
        offset += p*m
    return F_blocks, G_blocks

def build_companion_matrices(F_blocks, G_blocks, K, n_window):
    """
    Build the companion form matrices Ak, Bk, C for a MIMO system.
    F_blocks: list of n (p x p) matrices
    G_blocks: list of n (p x m) matrices
    K: (optional) feedback gain matrix (shape: m x (n*p)), or None
    Returns: Ak, Bk, C
    """
    n = n_window  # Using n_window since it represents the window size used throughout
    p = F_blocks[0].shape[0]
    m = G_blocks[0].shape[1]

    # Build Ak
    Ak = np.zeros((n*p, n*p))
    for i in range(n):
        Ak[i*p:(i+1)*p, 0:p] = -F_blocks[i]
        if i < n-1:
            Ak[i*p:(i+1)*p, (i+1)*p:(i+2)*p] = np.eye(p)
    # Add BK term if K is provided

    # Build Bk
    Bk = np.vstack(G_blocks)  # shape: (n*p, m)

    Ak += Bk @ K

    # Build C
    C = np.hstack([np.eye(p)] + [np.zeros((p, p)) for _ in range(n-1)])  # shape: (p, n*p)

    return Ak, Bk, C

def build_prediction_matrices(Ak, Bk, C, l):
    """
    Build Gamma_hat and T_hat for MPC prediction.
    Ak: (n_window*p, n_window*p)
    Bk: (n_window*p, m)
    C: (p, n_x)
    l: prediction horizon
    Returns: Gamma_hat (l*p, n_window*p), T_hat (l*p, l*m)
    """
    n_x = Ak.shape[0] # n_x = n_window * p
    m = Bk.shape[1]
    p = C.shape[0]
    
    # Initialize matrices with correct dimensions
    Gamma_hat = np.zeros((l*p, n_x))  # l*p rows for stacked outputs
    T_hat = np.zeros((l*p, l*m))      # l*m columns for stacked inputs
    
    # Build Gamma_hat (observability matrix)
    for i in range(l):
        Gamma_hat[i*p:(i+1)*p, :] = C @ np.linalg.matrix_power(Ak, i)
    
    H_blocks = []
    for j in range(l):
        H_j = C @ np.linalg.matrix_power(Ak, j) @ Bk
        H_blocks.append(H_j)
    
    for i in range(1, l):
        for j in range(i):
            block = H_blocks[i-j-1]  # H_{k, i-j}
            T_hat[i*p:(i+1)*p, j*m:(j+1)*m] = block
    
    return Gamma_hat, T_hat

def solve_mpc(Ak, Bk, C, C_t, y, tau, l, Q, ref_traj, Umin, Umax, F_blocks, G_blocks, R_rate=None, dUmin=None, dUmax=None):
    """
    Ak, Bk, C: system matrices
    x1k: current augmented state (n_x, 1) = y
    l: prediction horizon
    Q: (l*3, l*3) output tracking weight for controlled variables
    R: (l*m, l*m) input move weight
    ref_traj: (l*3, 1) stacked reference trajectory for controlled variables
    u_prev: (m, 1) previous input
    Umin, Umax: (l*m, 1) input constraints
    F_blocks: list of F matrices for prediction
    G_blocks: list of G matrices for prediction
    """
    n_x = Ak.shape[0]
    m = Bk.shape[1]
    p = C.shape[0]
    n_window = n_x // p  # Number of windows

    # Build prediction matrices
    Gamma_hat, T_hat = build_prediction_matrices(Ak, Bk, C, l)

    # Decision variable: future input sequence
    U = cp.Variable((l*m, 1))
     
    # Predicted output
    x_hat_k = np.zeros((n_x, 1))  
    
    # Handle both 1D and 2D input arrays
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Convert 1D to 2D column vector
    
    x_hat_k[:p, :] = y[:p, :]  # Copy first p rows from y
    
    if n_window > 1:
        for i in range(2, n_window):
            x_i_k = np.zeros(p, 1)
            for j in range (n_window-i+1):
                x_i_k = x_i_k - F_blocks[i+j-1] @ y[i*p:(i+1)*p] + G_blocks[i+j-1] @ tau[i*m:(i+1)*m]
                x_hat_k[:i*p] = x_i_k

    # Full output prediction
    x1k = Ak @ x_hat_k + Bk @ U[:m]  # Initial state prediction
    Y_pred = Gamma_hat @ x1k + T_hat @ U

    # Extract controlled variables using C_t
    C_t_block = np.kron(np.eye(l), C_t)  # Block diagonal for prediction horizon
    Y_pred_controlled = C_t_block @ Y_pred

    # Build input rate (ΔU) variables and constraints
    # ΔU[0] = U[0] - tau (current input - previous input)
    # ΔU[i] = U[i] - U[i-1] for i > 0
    if R_rate is not None:

        # Define control rate differences as expressions (not variables)
        dU_list = []
        
        # Handle both 1D and 2D tau arrays
        if tau.ndim == 1:
            tau_reshaped = tau.reshape(-1, 1)
        else:
            tau_reshaped = tau
            
        # First rate: ΔU[0] = U[0] - tau
        dU_list.append(U[:m] - tau_reshaped)
        
        # Subsequent rates: ΔU[i] = U[i] - U[i-1]
        for i in range(1, l):
            dU_list.append(U[i*m:(i+1)*m] - U[(i-1)*m:i*m])
        
        # Stack all rate differences
        dU = cp.vstack(dU_list)
        
        # Cost function with tracking, control, and control rate penalties
        cost = cp.quad_form(Y_pred_controlled - ref_traj, Q) + cp.quad_form(dU, R_rate)
        
        # Constraints on inputs and input rates
        constraints = [Umin <= U, U <= Umax]
        constraints.extend([dUmin <= dU, dU <= dUmax])
    else:
        # Cost function with tracking and control penalties
        cost = cp.quad_form(Y_pred_controlled - ref_traj, Q)
        
        # Constraints
        constraints = [Umin <= U, U <= Umax]

    # Solve QP with multiple fallback solvers
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    prob.solve(solver=cp.OSQP)
    U_opt = U.value
        
    return U_opt[:m]


def compute_partitioned_feedback_linearization(data, tau_k, contact_force):

    R1 = 1
    R2 = 1
    L1 = 2
    L2 = 2
    theta1 = 2 * np.arccos(data.qpos[3])
    theta2 = data.qpos[7]
    theta1_dot = -data.qvel[5]
    theta2_dot = data.qvel[6]
    x = data.qpos[0]  # Linear position in x direction
    z = data.qpos[2]  # Linear position in z direction
    x_dot = data.qvel[0]  # Linear velocity in x direction
    z_dot = data.qvel[2]  # Linear velocity in z direction
    ms = 1
    m1 = 1
    m2 = 1
    msc = 420
    I1 = 1/3
    I2 = 1/3

    q = np.array([theta1, theta2, x, z])  # State vector
    q_dot = np.array([theta1_dot, theta2_dot, x_dot, z_dot])  # State derivative vector

    A = np.array([[0, 0, 1, 0], 
                  [0, 0, 0, 1]])
    B = np.array([[-R1*np.sin(theta1), 0, 1, 0],
                  [R1*np.cos(theta1), 0, 0, 1]])
    C = np.array([[-L1*np.sin(theta1)-R2*np.sin(theta1+theta2), -R2*np.sin(theta1+theta2), 1, 0],
                  [L1*np.cos(theta1)+R2*np.cos(theta1+theta2), R2*np.cos(theta1+theta2), 0, 1]])
    D = np.array([[-L1*np.sin(theta1)-L2*np.sin(theta1+theta2), -L2*np.sin(theta1+theta2), 1, 0],
                 [L1*np.cos(theta1)+L2*np.cos(theta1+theta2), L2*np.cos(theta1+theta2), 0, 1]])
    E = np.array([1, 0, 0, 0]).reshape(1, 4)  # Ensure E is a column vector
    F = np.array([1, 1, 0, 0]).reshape(1, 4)  # Ensure F is a column vector

    M = ms * (A.T @ A) + m1 * (B.T @ B) + m2 * (C.T @ C) + msc * (D.T @ D) + I1 * (E.T @ E) + I2 * (F.T @ F)

    L = np.array([[1, 0],
                  [0, -1]]) #TODO: Figure out this matrix
    
    dMdq = np.array([[[[m2*((-2*L1*np.sin(theta1) - 2*R2*np.sin(theta1 + theta2))*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2)) + (-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2))*(-2*L1*np.cos(theta1) - 2*R2*np.cos(theta1 + theta2))) + msc*((-2*L1*np.sin(theta1) - 2*L2*np.sin(theta1 + theta2))*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)) + (-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2))*(-2*L1*np.cos(theta1) - 2*L2*np.cos(theta1 + theta2))), m2*(-R2*(-L1*np.cos(theta1) - R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2) - R2*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)) + msc*(-L2*(-L1*np.cos(theta1) - L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2) - L2*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)), -R1*m1*np.cos(theta1) + m2*(-L1*np.cos(theta1) - R2*np.cos(theta1 + theta2)) + msc*(-L1*np.cos(theta1) - L2*np.cos(theta1 + theta2)), -R1*m1*np.sin(theta1) + m2*(-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2)) + msc*(-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2))], [m2*(-R2*(-L1*np.cos(theta1) - R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2) - R2*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)) + msc*(-L2*(-L1*np.cos(theta1) - L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2) - L2*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)), 0, -L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), -L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2)], [-R1*m1*np.cos(theta1) + m2*(-L1*np.cos(theta1) - R2*np.cos(theta1 + theta2)) + msc*(-L1*np.cos(theta1) - L2*np.cos(theta1 + theta2)), -L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), 0, 0], [-R1*m1*np.sin(theta1) + m2*(-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2)) + msc*(-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2)), -L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2), 0, 0]]], [[[m2*(-2*R2*(-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - 2*R2*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)) + msc*(-2*L2*(-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - 2*L2*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)), m2*(-R2*(-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - R2*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)) + msc*(-L2*(-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - L2*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)), -L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), -L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2)], [m2*(-R2*(-L1*np.sin(theta1) - R2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - R2*(L1*np.cos(theta1) + R2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)) + msc*(-L2*(-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2))*np.cos(theta1 + theta2) - L2*(L1*np.cos(theta1) + L2*np.cos(theta1 + theta2))*np.sin(theta1 + theta2)), 0, -L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), -L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2)], [-L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), -L2*msc*np.cos(theta1 + theta2) - R2*m2*np.cos(theta1 + theta2), 0, 0], [-L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2), -L2*msc*np.sin(theta1 + theta2) - R2*m2*np.sin(theta1 + theta2), 0, 0]]], [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]])
    dMdq_column = np.column_stack([dMdq[:, :, k].flatten() for k in range(4)])  # shape (16, 4)

    D = dMdq_column.T @ np.kron(q_dot.reshape(4,1), np.eye(4)) - 0.5 * (np.kron(np.eye(4), q_dot.reshape(1,4)) @ dMdq_column)

    D1 = D[:2, :2]
    D2 = D[:2, 2:4]
    M1 = M[:2, :2]
    M2 = M[:2, 2:4]
    M4 = M[2:4, 2:4]

    q_dot = q_dot.reshape(4, 1)  # Ensure q_dot is a column vector
    tau_k = tau_k.reshape(2, 1)  # Ensure tau_k is a column vector
    vec = np.array([0, 1]).reshape(2, 1)  # Ensure vec is a column vector
    tau = np.linalg.inv(L) @ ( (D1-M1@np.linalg.inv(M2.T)@D2.T)@q_dot[:2] + D2@q_dot[2:] + M1@np.linalg.inv(M2.T)@vec*contact_force + (M2-M1@np.linalg.inv(M2.T)@M4)@tau_k )

    return tau