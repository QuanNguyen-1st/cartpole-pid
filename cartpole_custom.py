"""Point-mass cart-pole: an outer position PID and an inner angle PD.

Install: python -m pip install numpy matplotlib pillow gymnasium
Run:     python cartpole_pid.py
Animate: python cartpole_pid.py --render
Faster:  python cartpole_pid.py --render --speed 4
No I:    python cartpole_pid.py --render --no-i
No D:    python cartpole_pid.py --render --no-d
P only:  python cartpole_pid.py --render --no-i --no-d
GIF:     python cartpole_pid.py --gif cartpole.gif

The state is (p, p_dot, alpha, alpha_dot). Alpha is measured from upward,
positive when the pole leans right. This balances near upright, it is not
a swing-up controller. The default run starts at p=0.25 m, alpha=8 degrees.
This custom Gymnasium Env uses the point-mass model from the report, rather
than changing the discrete actions or rod physics of built-in CartPole-v1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import json
from pathlib import Path
import time as wall_time

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass(frozen=True)
class Parameters:
    M: float = 1.0          # Cart mass [kg]
    m: float = 0.1          # Point mass [kg]
    length: float = 0.5    # Pivot-to-mass distance [m]
    g: float = 9.81         # Gravity [m/s^2]
    dt: float = 0.01        # Sample period [s]
    force_limit: float = 10.0
    track_limit: float = 2.4
    angle_limit: float = np.pi / 3  # Stop if |alpha| > 60 degrees


# p''     = (u + m * sin(alpha)(l*a'^2 - g * cos(alpha)))/(M + m * sin^2(alpha))
# alpha'' = (g * sin(alpha) + p''*cos(alpha))/l.
def dynamics(state, force, par):
    """Exact nonlinear equations, with zero angle at upright."""
    p, velocity, alpha, omega = state
    s, c = np.sin(alpha), np.cos(alpha)
    denominator = par.M + par.m * s**2
    acceleration = (
        force + par.m * s * (par.length * omega**2 - par.g * c)
    ) / denominator
    angular_acceleration = (par.g * s - acceleration * c) / par.length
    return np.array([velocity, acceleration, omega, angular_acceleration])

# Runge–Kutta 4th order
def rk4(state, force, par):
    """Hold the applied force fixed during each sample interval."""
    h = par.dt
    k1 = dynamics(state, force, par)
    k2 = dynamics(state + h * k1 / 2, force, par)
    k3 = dynamics(state + h * k2 / 2, force, par)
    k4 = dynamics(state + h * k3, force, par)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def draw_scene(ax, state, par, time, artists=None):
    """Create the scene once, then update its artists on subsequent frames."""
    from matplotlib.patches import Rectangle

    if artists is None:
        ax.clear()
        ax.axhline(0, color="0.6", lw=1)
        ax.axvline(0, color="0.8", ls="--", lw=1)
        cart = Rectangle((-0.15, -0.09), 0.3, 0.16,
                         facecolor="#2563eb", edgecolor="black")
        ax.add_patch(cart)
        pole, = ax.plot([], [], color="#c2410c", lw=3)
        bob, = ax.plot([], [], "o", color="#c2410c", markersize=10)
        ax.set(xlim=(-par.track_limit - 0.2, par.track_limit + 0.2),
               ylim=(-0.2, max(0.85, par.length + 0.15)), xlabel="Cart position [m]")
        ax.set_aspect("equal", adjustable="box")
        artists = (cart, pole, bob, ax.set_title(""))
    p, _, alpha, _ = state
    tip = (p + par.length * np.sin(alpha), par.length * np.cos(alpha))
    cart, pole, bob, title = artists
    cart.set_x(p - 0.15)
    pole.set_data([p, tip[0]], [0, tip[1]])
    bob.set_data([tip[0]], [tip[1]])
    title.set_text(f"t = {time:.2f} s    pole angle = {np.rad2deg(alpha):.2f} deg")
    return artists


class PointMassCartPole(gym.Env):
    """Continuous force, exact point-mass physics, and Gymnasium's step API."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, par=None, duration=15.0, render_mode=None, bias=0.0,
                 playback_speed=1.0):
        super().__init__()
        self.par = Parameters() if par is None else par
        if min(self.par.M, self.par.m, self.par.length, self.par.dt,
               self.par.force_limit, duration) <= 0:
            raise ValueError("Masses, length, dt, force limit and duration must be positive.")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode.")
        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("Playback speed must be finite and positive.")
        self.render_mode = render_mode
        self.playback_speed = float(playback_speed)
        self.metadata = dict(self.metadata)
        # Physics still runs at every dt. Only displayed frames are skipped.
        self._frame_stride = max(1, int(np.ceil(
            self.playback_speed / (self.metadata["render_fps"] * self.par.dt))))
        self._wall_start = None
        self.max_steps = int(np.ceil(duration / self.par.dt))
        self.bias = float(bias)  # Constant external horizontal force [N]
        self.action_space = spaces.Box(-self.par.force_limit, self.par.force_limit,
                                       shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)
        self.state = None
        self.steps = 0
        self.ended = False
        self.figure = self.axes = None
        self._artists = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        initial = options.get("state", self.np_random.uniform(-0.05, 0.05, 4))
        self.state = np.asarray(initial, dtype=np.float64).copy()
        if self.state.shape != (4,) or not np.all(np.isfinite(self.state)):
            raise ValueError("Initial state must contain four finite numbers.")
        self.steps, self.ended = 0, False
        self._wall_start = None
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    def step(self, action):
        if self.state is None or self.ended:
            raise RuntimeError("Call reset() before starting or restarting an episode.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite array of shape (1,).")
        force = float(np.clip(action[0], -self.par.force_limit, self.par.force_limit))
        self.state = rk4(self.state, force + self.bias, self.par)
        self.steps += 1
        p, _, alpha, _ = self.state
        terminated = bool(abs(p) > self.par.track_limit or abs(alpha) > self.par.angle_limit)
        truncated = bool(self.steps >= self.max_steps)
        self.ended = terminated or truncated
        reward = float(-(p**2 + 10 * alpha**2 + 0.001 * force**2))
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), reward, terminated, truncated, {"force": force}

    def render(self):
        if self.render_mode is None or self.state is None:
            return None
        if (self.render_mode == "human" and not self.ended
                and self.steps % self._frame_stride != 0):
            return None
        import matplotlib.pyplot as plt
        if self.figure is None:
            self.figure, self.axes = plt.subplots(figsize=(9, 3))
            if self.render_mode == "human":
                plt.show(block=False)
        simulated_time = self.steps * self.par.dt
        self._artists = draw_scene(self.axes, self.state, self.par, simulated_time,
                                   self._artists)
        self.figure.canvas.draw()
        if self.render_mode == "rgb_array":
            return np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy()
        if self._wall_start is None:
            self._wall_start = wall_time.perf_counter() - simulated_time / self.playback_speed
        target = self._wall_start + simulated_time / self.playback_speed
        # Compensate for drawing time instead of adding a fixed delay each step.
        plt.pause(max(0.001, target - wall_time.perf_counter()))
        return None

    def close(self):
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = self.axes = None
            self._artists = None
            self._wall_start = None


@dataclass(frozen=True)
class Gains:
    angle_p: float
    angle_d: float
    position_p: float   # Coefficients in u = K*alpha + D*omega + kp*p + kd*v + ki*z
    position_d: float
    position_i: float


def design_gains(par, rates=(1.0, 1.2, 2.0, 3.0, 4.0)):
    """Match the nominal continuous-force rod model to five chosen poles."""
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (5,) or np.any(rates <= 0):
        raise ValueError("Choose five positive pole rates.")
    _, a1, a2, a3, a4, a5 = np.poly(-rates)
    M, m, l, g = par.M, par.m, par.length, par.g
    ki = M * l * a5 / g
    kp = M * l * a4 / g
    kd = (M * l * a3 + l * ki) / g
    D = M * l * a1 + l * kd
    K = M * l * a2 + (M + m) * g + l * kp
    return Gains(K, D, kp, kd, ki)


class CascadedPID:
    """Position PID -> reference lean; angle PD -> force.

    Position error e_p = -p; position integral I_p = -z.
    alpha_ref = -(kp*p + kd*v + ki*z)/K.
    u_raw = K*(alpha - alpha_ref) + D*omega.
    """
    def __init__(self, par, gains):
        self.par, self.gains = par, gains
        self.z = 0.0

    def reset(self):
        self.z = 0.0

    def action(self, state):
        p, v, alpha, omega = state
        k = self.gains
        alpha_ref = -(k.position_p * p + k.position_d * v + k.position_i * self.z) / k.angle_p
        raw = k.angle_p * (alpha - alpha_ref) + k.angle_d * omega
        force = float(np.clip(raw, -self.par.force_limit, self.par.force_limit))
        # Integrate after computing u: explicit Euler, consistent with the report.
        would_push_further = (raw - force) * (k.position_i * p) > 0
        if k.position_i != 0 and not would_push_further:
            self.z += self.par.dt * p
        return np.array([force], dtype=np.float64)


def simulate(par, initial, duration=15.0, render=False, bias=0.0, playback_speed=1.0,
             no_i=False, no_d=False):
    env = PointMassCartPole(par, duration, "human" if render else None, bias, playback_speed)
    gains = design_gains(par)
    # disable terms without retuning the remaining gains.
    if no_i:
        gains = replace(gains, position_i=0.0)
    if no_d:
        gains = replace(gains, angle_d=0.0, position_d=0.0)
    controller = CascadedPID(par, gains)
    state, _ = env.reset(seed=None, options={"state": initial})
    states, forces = [state.copy()], []
    terminated = False
    try:
        while True:
            action = controller.action(state)
            state, _, terminated, truncated, info = env.step(action)
            forces.append(info["force"])
            states.append(state.copy())
            if terminated or truncated:
                break
    finally:
        env.close()
    time = np.arange(len(states)) * par.dt
    return time, np.asarray(states), np.asarray(forces), terminated, gains


def save_plots(time, states, forces, destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True, layout="constrained")
    axes[0].plot(time, states[:, 0], color="#2563eb")
    axes[0].set_ylabel("Cart position [m]")
    axes[1].plot(time, np.rad2deg(states[:, 2]), color="#c2410c")
    axes[1].set_ylabel("Pole angle [deg]")
    axes[2].step(time[:-1], forces, where="post", color="#047857")
    axes[2].set(ylabel="Control force [N]", xlabel="Time [s]")
    for ax in axes:
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_gif(time, states, par, destination, playback_speed=1.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("Playback speed must be finite and positive.")
    fig, ax = plt.subplots(figsize=(8, 2.7))
    stride = max(1, round(playback_speed / (25 * par.dt)))
    frames = np.unique(np.r_[np.arange(0, len(time), stride), len(time) - 1]).tolist()
    artists = draw_scene(ax, states[0], par, time[0])
    fps = playback_speed / (stride * par.dt)
    animation = FuncAnimation(
        fig, lambda i: draw_scene(ax, states[i], par, time[i], artists),
        frames=frames, interval=1000 / fps)
    animation.save(destination, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Show animation.")
    parser.add_argument("--no-i", action="store_true",
                        help="Disable position integral.")
    parser.add_argument("--no-d", action="store_true",
                        help="Disable both derivative terms.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Live/GIF playback multiplier; e.g. 4 plays four times faster.")
    parser.add_argument("--save-gif", type=Path, help="Save a GIF.")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--angle", type=float, default=8.0, help="Initial angle in degrees.")
    parser.add_argument("--position", type=float, default=0.25, help="Initial position in metres.")
    parser.add_argument("--bias", type=float, default=0.0, help="Constant external force in newtons.")
    parser.add_argument("--outdir", type=Path, default=Path("cartpole_results"))
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive.")
    par = Parameters()
    initial = [args.position, 0.0, np.deg2rad(args.angle), 0.0]
    time, states, forces, failed, gains = simulate(
        par, initial, args.seconds, args.render, args.bias, args.speed,
        no_i=args.no_i, no_d=args.no_d)
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_plots(time, states, forces, args.outdir / "cartpole_response.png")
    # A force is defined on an interval; the last state has no subsequent force.
    data = np.column_stack((time, states, np.r_[forces, np.nan]))
    np.savetxt(args.outdir / "trajectory.csv", data, delimiter=",", comments="",
               header="time_s,p_m,velocity_m_s,alpha_rad,omega_rad_s,force_N")
    summary = {
        "parameters": asdict(par), "gains": asdict(gains),
        "controller": {"integral_enabled": not args.no_i,
                       "derivative_enabled": not args.no_d},
        "duration_s": float(time[-1]), "terminated_by_failure": failed,
        "final_state": states[-1].tolist(),
        "max_abs_position_m": float(np.max(np.abs(states[:, 0]))),
        "max_abs_angle_deg": float(np.rad2deg(np.max(np.abs(states[:, 2])))),
        "max_abs_force_N": float(np.max(np.abs(forces))),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.save_gif:
        args.save_gif.parent.mkdir(parents=True, exist_ok=True)
        save_gif(time, states, par, args.save_gif, args.speed)
    if failed:
        reason = ("Removing I or D may destabilize the controller."
                  if args.no_i or args.no_d else "Try a smaller initial tilt.")
        raise SystemExit("Balance failed: cart or angle limit exceeded. " + reason)


if __name__ == "__main__":
    main()
