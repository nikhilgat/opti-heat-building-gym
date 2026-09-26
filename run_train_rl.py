"""
run_train_rl.py – Train PPO, SAC, DDPG, TD3, or A2C on LLEC-HeatPumpHouse environments.
This script sets up parallel training and evaluation environments with configurable observation
and reward settings. It trains an RL agent using Stable Baselines3 and saves the model and logs.
"""

import os
import sys
import time
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Trigger registration of the custom Gym IDs
from llec_building_gym import BaseBuildingGym
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("sys.argv: %s", sys.argv)


# Helper classes & functions
class SeedWrapper(gym.Wrapper):
    """Reproducible but different seed on every reset: seed, seed+10000, seed+20000, ...
    (a fixed seed on every reset replays the exact same day/weather each episode)."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._n_resets = 0

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self._seed + 10_000 * self._n_resets
        self._n_resets += 1
        return super().reset(seed=seed, options=options)


def make_env(env_id: str, rank: int, base_seed: int, max_episode_steps: int = 288, **env_kwargs):
    """Factory function for SubprocVecEnv."""

    def _init() -> gym.Env:
        env = gym.make(env_id, max_episode_steps=max_episode_steps, **env_kwargs)
        env = SeedWrapper(env, seed=base_seed + rank)
        return env

    return _init


def select_model(algorithm: str, env: gym.Env, seed: int, batch_size: int = 64, buffer_size: int = 100_000, n_steps: int = 288, tensorboard_log: str = None, gamma: float = 0.99):
    """
    Selects and configures a model for training based on the algorithm name.
    """
    # small MLPs: on-policy PPO/A2C are faster on CPU (SB3 recommendation), off-policy on GPU
    device = torch.device("cuda" if torch.cuda.is_available() and algorithm in ("sac", "ddpg", "td3") else "cpu")
    logger.info("Training on device: %s", device)

    # Hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    episode_length = 288
    learning_starts = 10 * episode_length
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    elif algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device,
            use_sde=True,
            tensorboard_log=tensorboard_log,
        )

    elif algorithm == "ddpg":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            train_freq=1,  # SB3 default (1, "episode") only works with a single env
            gradient_steps=1,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    elif algorithm == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=learning_starts,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            train_freq=1,  # SB3 default (1, "episode") only works with a single env
            gradient_steps=1,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    elif algorithm == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=n_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device,
            use_sde=True,
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return model


# Main
def main():
    """Parse CLI arguments and run training for selected algorithms."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", default="ppo", choices=["ppo", "sac", "ddpg", "td3", "a2c"]
    )
    parser.add_argument("--vec-env", default="dummy", choices=["dummy", "subproc"], dest="vec_env",
                        help="dummy = envs in the training process (fast env, no IPC); subproc = one process per env.")
    parser.add_argument("--torch-threads", type=int, default=1, dest="torch_threads",
                        help="CPU threads for PyTorch in this run (avoid oversubscription when running several runs).")
    parser.add_argument("--economic-weight", type=float, default=1.0, dest="economic_weight",
                        help="Weight of the energy-cost term in the reward (band penalty stays 10/K).")
    parser.add_argument("--tag", default="", help="Suffix for the model/log folder, e.g. ew5 -> models/<mode>/<obs>_ew5/.")
    parser.add_argument("--start-mode", default="random", choices=["random", "carry"], dest="start_mode",
                        help="Training episode start temperature: random 16-23 degC, or carry = where the previous episode ended.")
    parser.add_argument("--eval-T-in", type=float, default=20.5, dest="eval_T_in",
                        help="Start temperature of every evaluation episode (fixed, so evaluations are comparable).")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor. ~0.998 (~40 h horizon) suits the building's ~56 h time constant.")
    parser.add_argument("--backup-kw", type=float, default=0.0, dest="backup_kw",
                        help="Electric backup heater (kW, COP 1) next to the heat pump (IDM AERO ALM 4-12: 6 kW).")
    parser.add_argument("--forecast-error-path", default=None, dest="forecast_error_path",
                        help="Real day-ahead weather forecast errors (prepare_langenhagen_data.py --forecast), used by C06.")
    parser.add_argument("--eval-weeks", type=int, default=20, dest="eval_weeks",
                        help="Fixed held-out episodes per evaluation (same ones every time); rounded to a multiple of --num-envs.")
    parser.add_argument(
        "--episode-days", type=int, default=7, dest="episode_days",
        help="Episode length in days. Multi-day episodes make the agent live with a cold house "
             "(the building's time constant is ~56 h, so one-day episodes let it coast).",
    )
    parser.add_argument("--timesteps", type=float, default=1e6)
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument(
        "--reward_mode", default="temperature", choices=["temperature", "combined", "band"],
        help="band = real-building rules: heating only, Oct-Apr, comfort band 19-22 degC",
    )
    parser.add_argument("--energy-price-path", default="data/langenhagen_price_2025.csv")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Use training split of price data (else test split)",
    )
    parser.add_argument(
        "--obs_variant",
        default="T01",
        choices=["T01", "T02", "T03", "T04", "C01", "C02", "C03", "C04", "C05", "C06"],
    )
    parser.add_argument("--mC", type=float, default=300, help="Thermal capacitance (J/K).")
    parser.add_argument("--K", type=float, default=20, help="Heat loss coefficient (W/K).")
    parser.add_argument("--Q_HP_Max", type=float, default=1500, help="Max heat pump power (W).")
    parser.add_argument("--cop_heat", type=float, default=1.0, help="Fixed heating COP (ignored if --scop is set).")
    parser.add_argument("--cop_cool", type=float, default=1.0, help="Fixed cooling COP.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (bigger = better GPU utilization).")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size (SAC/DDPG/TD3 only).")
    parser.add_argument(
        "--outdoor-temperature-path",
        default=None,
        help="Optional real outdoor temperature CSV (else synthetic profile is used).",
    )
    parser.add_argument(
        "--dt-scale",
        type=float,
        default=1e-3,
        dest="dt_scale",
        help=(
            "Thermal integration scale factor in update_Tin. Default 1e-3 reproduces "
            "the repo's original toy-building calibration (mC=300, K=20). Pass 1.0 "
            "when mC/K/Q_HP_Max are true SI values (J/K, W/K, W), e.g. for a real "
            "building, so the resulting thermal time constant (mC/K seconds) is physically correct."
        ),
    )
    parser.add_argument(
        "--scop",
        action="store_true",
        help=(
            "Use the full IDM AERO ALM 4-12 heat pump model (weather-compensated "
            "flow temperature + 2D T_air/T_flow COP interpolation with duty-cycle "
            "averaging) instead of the fixed --cop_heat value."
        ),
    )
    parser.add_argument(
        "--tensorboard-log",
        default=None,
        dest="tensorboard_log",
        help=(
            "Directory for TensorBoard logs (e.g. 'tb_logs'). Omit to disable. "
            "View live with: tensorboard --logdir <dir>, then open http://localhost:6006"
        ),
    )
    args = parser.parse_args()

    args.timesteps = int(args.timesteps)
    torch.set_num_threads(args.torch_threads)
    sub = args.obs_variant + (f"_{args.tag}" if args.tag else "")
    VecEnv = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    logger.info("Parsed arguments: %s", vars(args))

    # Map usecase to environment Gym Environment ID alias per reward mode (registered by llec_building_gym)
    env_id = {
        "temperature": "LLEC-HeatPumpHouse-1R1C-Temperature-v0",
        "combined": "LLEC-HeatPumpHouse-1R1C-Combined-v0",
        "band": "LLEC-HeatPumpHouse-1R1C-Band-v0",
    }[args.reward_mode]

    # Training environments setup
    train_env = VecEnv(
        [
            make_env(
                env_id,
                rank=i,
                base_seed=args.seed,
                energy_price_path=args.energy_price_path,
                training=args.training,
                schedule_type=None,
                obs_variant=args.obs_variant,
                mC=args.mC,
                K=args.K,
                Q_HP_Max=args.Q_HP_Max,
                cop_heat=args.cop_heat,
                cop_cool=args.cop_cool,
                outdoor_temperature_path=args.outdoor_temperature_path,
                dt_scale=args.dt_scale,
                use_scop=args.scop,
                simulation_time=args.episode_days * 24 * 3600,
                max_episode_steps=args.episode_days * 288,
                backup_kw=args.backup_kw,
                economic_weight=args.economic_weight,
                start_mode=args.start_mode,
                forecast_error_path=args.forecast_error_path,
            )
            for i in range(args.num_envs)
        ]
    )
    train_env = VecMonitor(train_env)

    # Evaluation environments setup
    eval_env = VecEnv(
        [
            make_env(
                env_id,
                rank=i,
                # ensure no overlap with training seeds
                base_seed=args.seed + 10_000,
                eval_subset=(i, args.num_envs, max(args.eval_weeks // args.num_envs, 1)),
                fixed_T_in=args.eval_T_in,
                energy_price_path=args.energy_price_path,
                training=False,
                schedule_type=None,
                obs_variant=args.obs_variant,
                mC=args.mC,
                K=args.K,
                Q_HP_Max=args.Q_HP_Max,
                cop_heat=args.cop_heat,
                cop_cool=args.cop_cool,
                outdoor_temperature_path=args.outdoor_temperature_path,
                dt_scale=args.dt_scale,
                use_scop=args.scop,
                simulation_time=args.episode_days * 24 * 3600,
                max_episode_steps=args.episode_days * 288,
                backup_kw=args.backup_kw,
                economic_weight=args.economic_weight,
                forecast_error_path=args.forecast_error_path,
            )
            for i in range(args.num_envs)
        ]
    )
    eval_env = VecMonitor(eval_env)

    # Evaluation callback setup
    eval_callback = EvalCallback(
        eval_env,
        # seed in the folder so several seeds of one algorithm don't overwrite each other's best model
        best_model_save_path=f"models/{args.reward_mode}/{sub}/best_{args.algorithm}_seed{args.seed}/",
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.num_envs * max(args.eval_weeks // args.num_envs, 1),
        deterministic=True,
        render=False,
    )
    # Training Model
    model = select_model(
        args.algorithm, train_env, args.seed,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        tensorboard_log=args.tensorboard_log, gamma=args.gamma,
    )
    logger.info("Observation space: %s", train_env.observation_space)
    logger.info("Action space:      %s", train_env.action_space)
    if args.tensorboard_log:
        logger.info(
            "TensorBoard logging to: %s (view live with: tensorboard --logdir %s)",
            args.tensorboard_log, args.tensorboard_log,
        )
    t0 = time.time()
    tb_log_name = f"{args.algorithm}_{args.reward_mode}_{sub}_seed{args.seed}"
    model.learn(
        total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True,
        tb_log_name=tb_log_name,
    )
    # Log training end time, duration, and save location
    logger.info("Training completed in %.2f min", (time.time() - t0) / 60)

    # Saving Model
    save_dir = f"models/{args.reward_mode}/{sub}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.algorithm}_model_seed{args.seed}"
    model.save(save_path)
    logger.info("Model saved to: %s", save_path)
    logger.info(
        f"Best model path: models/{args.reward_mode}/{sub}/best_{args.algorithm}_seed{args.seed}/"
    )

    # Close environments
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

# python run_train_rl.py --algorithm ppo --reward_mode temperature --obs_variant T01 --training --seed 18 --timesteps 1e6
