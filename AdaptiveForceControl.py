import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import controllers as rc
import os
import glfw
import time


# --- Parameters ---
Fd_z = 25      # Desired normal force (N)
descent_vz = -0.1  # Descent speed (m/s)
ascent_vz = +0.1   # Ascent speed (m/s)
sampling_time = 2.0            # seconds


xml_path = Path(__file__).parent / "models/2Dspacecraft_fromEE.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
model.opt.timestep = 0.0005 
lamba_value = 0.05 # Regularization parameter for LMI


# Controller gains
# K_matrix = rc.compute_gain_lmi_paper_version(lambda_val=lamba_value, Ts=model.opt.timestep)
# print(K_matrix)

Kp = 1.5  # Proportional gain using LMI intuition
K_matrix = np.array([
    [1.1, 0.1, 0.3],  
    [1.1, -1.1, -0.3]
]) * Kp

model.opt.iterations = 50
model.opt.tolerance = 1e-8
model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)

# Set initial linear velocity (vx, vy, vz)
data.qvel[0] = 0.0  # vx
data.qvel[1] = 0.0  # vy
data.qvel[2] = descent_vz  # initial descent speed

# Set initial joint angles (in radians)
data.qpos[3:7] = [np.cos(np.pi/8), 0, np.sin(np.pi/8), 0]  # 45 deg quaternion [w,x,y,z]
data.qpos[7] = np.pi/2  # tau1

# Initialize lists to store force data
force_history = []
friction_history = []
total_force_history = []
time_history = []
error_history = []
vx_history = []
vz_history = []
sim_time = 0.0

# Simulation parameters
duration = 5.0  # seconds
n_steps = int(duration / model.opt.timestep)

# State machine variables
phase = 'descent'
contact_start_time = None

# Parameters for RLS and PCAC
tau_n = 40
tau_d = 200
eta = 0.1
alpha = 0.001
n_window = 1  # Number of samples for identification
p = 5 # Number of outputs 
m = 2 # Number of inputs 

# MPC parameters
l = 50
Q = np.kron(np.eye(l), np.diag([1000, 100, 1])) 
R_rate = np.eye(l*m) 
C_t = np.hstack([np.zeros((3,2)), np.eye(3)])
ref_traj = np.zeros((3*l,1))
u_max = 100  
u_min = -100
U_max = np.kron(np.ones((l*m,1)), u_max)
U_min = np.kron(np.ones((l*m,1)), u_min)

# Control rate limits
du_max = 10
du_min = -10
dU_max = np.kron(np.ones((l*m,1)), du_max)
dU_min = np.kron(np.ones((l*m,1)), du_min)

# Initial values for RLS
P_0 = 10*np.eye(n_window*p*(p+m))
w_0 = np.zeros((n_window*p*(p+m), 1))

P_k = P_0.copy()
w_k = w_0.copy()
tau_k = np.zeros((2, 1))  # Initial control torques

epsilon_buffer = []
tau_buffer = []

# Real-time factor calculation variables
wall_start_time = time.time()
step_start_time = time.time()

ctrl_history_arr = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    
    for i in range(n_steps):
        step_start_time = time.time()
        
        # Get contact forces for logging/plotting
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, contact_force)
        normal_force = contact_force[0]
        friction_force = contact_force[1:3]
        total_force = np.array([friction_force[0], friction_force[1], normal_force])
        in_contact = abs(normal_force) > 0

        if phase == 'descent':
            tau = np.zeros(2)  # No control torques during descent
            data.ctrl[:] = tau
            e = -Fd_z  
            if in_contact:
                phase = 'contact'
                contact_start_time = sim_time
                print(f"Contact detected at t = {sim_time:.3f} s")

        elif phase == 'contact':
            vx = data.qvel[0]
            vz = data.qvel[2]
            e = normal_force - Fd_z  
            theta_1_dot = -data.qvel[5]  # Angular velocity of joint 1 ("tau1")
            theta_2_dot = data.qvel[6]  # Angular velocity of joint 2 ("tau2")
            chi_k = np.array([vx, vz, e])
            epsilon_k = np.array([theta_1_dot, theta_2_dot, vx, vz, e])
            tau_k_robust = K_matrix @ chi_k
            
            # PCAC
            t = i  # Current time step
            y = epsilon_k
            P_k_new, w_k_new = rc.pcac_rls_update_mimo(t, y, tau_k, P_k, w_k, p, m, n_window, tau_n, tau_d, eta, alpha)
            P_k = P_k_new
            w_k = w_k_new
            F_blocks, G_blocks = rc.extract_F_G_from_wk(w_k_new, p, m, n_window)
            K = np.hstack([K_matrix, np.zeros((2,2))])
            Ak, Bk, C = rc.build_companion_matrices(F_blocks, G_blocks, K, n_window)
            tau_k_aug = rc.solve_mpc(Ak, Bk, C, C_t, y, tau_k, l, Q, ref_traj, U_min, U_max, F_blocks, G_blocks, R_rate, dU_min, dU_max)

            # Sum and assign
            tau_k_robust = tau_k_robust.reshape(2, 1)  # Ensure tau_k_robust is a column vector
            tau_k_aug = tau_k_aug.reshape(2, 1)  # Ensure tau_k_aug is a column vector
            tau_k = tau_k_robust + tau_k_aug

            tau = rc.compute_partitioned_feedback_linearization(data, tau_k, normal_force) #TODO: WATCH THIS LINE, IF YOU TAKE THE FEEDBACK LINEARIZATION OFF THE CONTROLLER WORKS
            data.ctrl[:] = tau[:, 0]

            if (sim_time - contact_start_time) >= sampling_time:
                phase = 'ascent'
                print(f"Ascent initiated at t = {sim_time:.3f} s")
        elif phase == 'ascent':
            tau = np.zeros(2)
            data.qvel[2] = ascent_vz
            data.ctrl[:] = tau
            e = -Fd_z

        # Record forces and control/velocities
        error_history.append(e)
        force_history.append(normal_force)
        friction_history.append(friction_force)
        time_history.append(sim_time)
        vx_history.append(data.qvel[0])
        vz_history.append(data.qvel[2])
        sim_time += model.opt.timestep
        ctrl_history_arr.append(np.copy(data.ctrl))  # Store a copy at each step

        viewer.sync()
        mujoco.mj_step(model, data)
        
        # Proper real-time control: Calculate elapsed time for this step
        step_elapsed_time = time.time() - step_start_time
        time_until_next_step = model.opt.timestep - step_elapsed_time
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Calculate and display real-time factor
wall_end_time = time.time()
total_wall_time = wall_end_time - wall_start_time
total_sim_time = duration
real_time_factor = total_sim_time / total_wall_time

print(f"\n--- Real-Time Factor Analysis ---")
print(f"Simulation time: {total_sim_time:.3f} seconds")
print(f"Wall clock time: {total_wall_time:.3f} seconds")
print(f"Real-time factor: {real_time_factor:.3f}x")
print(f"Timestep: {model.opt.timestep} seconds")
print(f"Total steps: {n_steps}")
print(f"Average step computation time: {(total_wall_time/n_steps)*1000:.2f} ms")

if real_time_factor > 1.0:
    print(f"Simulation runs {real_time_factor:.2f}x faster than real-time")
elif real_time_factor < 1.0:
    print(f"Simulation runs {1/real_time_factor:.2f}x slower than real-time")
else:
    print("Simulation runs at exactly real-time")

# --- Plotting ---
# Convert list to numpy array for proper indexing
ctrl_history_arr = np.array(ctrl_history_arr)

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(time_history, force_history, 'g-', label='Contact Force (N)')
plt.axhline(y=Fd_z, color='r', linestyle='--', label='Desired Force')
plt.ylabel('Contact Force (N)')
plt.title('Contact Force vs Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_history, ctrl_history_arr[:, 0], 'b-', label='Tau1')
plt.plot(time_history, ctrl_history_arr[:, 1], 'r-', label='Tau2')
plt.ylabel('Torque (N⋅m)')
plt.title('Joint Torques vs Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_history, vx_history, 'b-', label='Vx')
plt.plot(time_history, vz_history, 'r-', label='Vz')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocities vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


