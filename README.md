# Adaptive Force Control for Spacecraft Manipulation (MuJoCo)

This project implements an adaptive force control for a 2D spacecraft manipulator in a microgravity environment, using the MuJoCo physics engine. The main focus is on robust force control during contact with an unknown surface, using advanced control techniques such as robust feedback, recursive least squares (RLS) identification, model predictive control (MPC), and partitioned feedback linearization.

## Main Components

- **AdaptiveForceControl.py**: Main simulation script. Runs a MuJoCo simulation of a 2D spacecraft manipulator, applying adaptive force control with online system identification, MPC, robust feedback, and feedback linearization control to regulate contact force during surface interaction. Includes plotting and real-time analysis.
- **controllers.py**: Contains control algorithms, including gain computation via LMI, RLS parameter identification, MPC setup and solution, and feedback linearization routines.
- **models/2Dspacecraft_fromEE.xml**: MuJoCo XML model of the 2D spacecraft manipulator, including joints, links, actuators, sensors, and a contact surface.

## Setup Instructions

### 1. Install Python Dependencies

It is recommended to use a virtual environment (e.g., `venv` or `conda`).

```bash
pip install -r requirements.txt
```

### 2. Install MuJoCo

- Install MuJoCo (>=2.3.0) following the [official MuJoCo installation guide](https://mujoco.readthedocs.io/en/stable/).

### 3. Additional Notes
- The simulation uses the MuJoCo viewer for visualization. 
- If you encounter issues with the viewer or MuJoCo installation, consult the [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/).

## Usage

To run the main simulation and plot results:

```bash
python AdaptiveForceControl.py
```

This will launch the MuJoCo viewer, run the simulation, and display plots of contact force, joint torques, and velocities.

## File Overview

- `AdaptiveForceControl.py`: Main entry point. Handles simulation loop, state machine (descent/contact/ascent), data logging, and plotting.
- `controllers.py`: Implements control logic (LMI gain, RLS, MPC, feedback linearization).
- `models/2Dspacecraft_fromEE.xml`: MuJoCo model of the spacecraft manipulator and environment.
- `requirements.txt`: Python dependencies.
- `dMdq_symbolic.py`: Symbolic calculation of $\frac{\partial M}{\partial q}$ needed for the feedback linearization as shown in [[1](#references)]

## References
[1] N. Mohseni, D. S. Bernstain, M.B. Quadrelli, *Adaptive Force-Control Augumentation for Small Celestial Body Sampling*, Journal of Guidance, Control, and Dynamics, Vol. 46, No. 12, December 2023, https://doi.org/10.2514/1.G007575
- [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/)
- [cvxpy Documentation](https://www.cvxpy.org/)
- [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [SciPy](https://scipy.org/)

---

For questions or issues, please contact me at pierluigi.vergari@polito.it.
