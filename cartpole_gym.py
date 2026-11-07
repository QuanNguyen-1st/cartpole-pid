"""
Install: python -m pip install "gymnasium[classic-control]" pillow
Run:     python cartpole_gym_pid.py --render
Record:  python cartpole_gym_pid.py --save-gif cartpole.gif
Compare: python cartpole_gym_pid.py --no-i --episodes 20

The cascade PID computes a requested force. Its sign selects one of
Gymnasium's two actions.
"""

import argparse
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image


def design_gains(plant, rates=(1.0, 1.2, 2.0, 3.0, 4.0)):
    """Match the nominal continuous-force rod model to five chosen poles."""
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (5,) or np.any(rates <= 0):
        raise ValueError("Choose five positive pole rates.")
    cart_mass = plant.masscart
    pole_mass = plant.masspole
    length = plant.length
    gravity = plant.gravity
    c = 4.0 / 3.0

    # polynomial coefficients
    _, a1, a2, a3, a4, a5 = np.poly(-rates)

    # alpha = 4/3 * (M+m)-m
    alpha = c * (cart_mass + pole_mass) - pole_mass
    leading = alpha * length

    position_i = leading * a5 / gravity
    position_p = leading * a4 / gravity
    position_d = (leading * a3 + c * length * position_i) / gravity
    angle_d = leading * a1 + c * length * position_d
    angle_p = (
        leading * a2
        + (cart_mass + pole_mass) * gravity
        + c * length * position_p
    )
    return angle_p, angle_d, position_p, position_d, position_i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", action="store_true", help="Show Gymnasium's animation.")
    display.add_argument("--save-gif", type=Path, metavar="FILE", help="Save a GIF.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the first episode.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--steps", type=int, help="Override Gymnasium's episode step limit.")
    parser.add_argument("--no-i", action="store_true", help="Disable the position integral.")
    parser.add_argument("--no-d", action="store_true", help="Disable both derivative terms.")
    args = parser.parse_args()

    if args.seed < 0 or args.episodes < 1 or (args.steps is not None and args.steps < 1):
        parser.error("Use a nonnegative seed and positive episode and step counts.")

    render_mode = None
    if args.save_gif is not None:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"

    env = gym.make(
        "CartPole-v1",
        render_mode=render_mode,
        max_episode_steps=args.steps,
    )
    try:
        plant = env.unwrapped
        angle_p, angle_d, position_p, position_d, position_i = design_gains(plant)
        if args.no_i:
            position_i = 0.0
        if args.no_d:
            angle_d = 0.0
            position_d = 0.0
        force_limit = plant.force_mag
        completed = 0

        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed)
            integral = 0.0
            steps = 0
            peak_angle = abs(float(observation[2]))

            frames = []
            if args.save_gif is not None:
                frames.append(Image.fromarray(env.render()))

            while True:
                position, velocity, angle, angular_velocity = map(float, observation)
                requested_force = (
                    angle_p * angle
                    + angle_d * angular_velocity
                    + position_p * position
                    + position_d * velocity
                    + position_i * integral
                )

                # anti-windup before choosing a binary action.
                clipped_force = min(force_limit, max(-force_limit, requested_force))
                pushes_further = (requested_force - clipped_force) * position_i * position > 0
                if position_i and not pushes_further:
                    integral += plant.tau * position

                # -10 N for 0 and +10 N for 1.
                action = int(clipped_force >= 0.0)
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                peak_angle = max(peak_angle, abs(float(observation[2])))

                if args.save_gif is not None:
                    frames.append(Image.fromarray(env.render()))

                if terminated or truncated:
                    break

            if args.save_gif is not None:
                gif_path = args.save_gif
                if args.episodes > 1:
                    gif_path = gif_path.with_name(f"{gif_path.stem}_seed{seed}.gif")

                frames[0].save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=round(1000 * plant.tau),  # milliseconds per frame.
                    loop=0,
                )
                print(f"Saved GIF to {gif_path}")

            outcome = "terminated" if terminated else "time limit reached"
            completed += int(truncated and not terminated)
            print(
                f"Seed {seed}: {steps} steps ({steps * plant.tau:.2f} s), "
                f"peak tilt {math.degrees(peak_angle):.2f} deg, {outcome}."
            )

        print(f"Completed without losing balance: {completed}/{args.episodes} episodes.")
    finally:
        env.close()


if __name__ == "__main__":
    main()