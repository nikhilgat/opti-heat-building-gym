# File: llec_building_gym/controllers/mpc_controller.py
import os

import numpy as np
import pyomo.environ as pyo

# Smoothing constant for |action| so that ipopt stays differentiable at 0
PRICE_ABS_EPS = 1e-6


# MPCController
class MPCController:
    """
    A simplified Model Predictive Controller (MPC) that can optimize either
    temperature objectives alone or a combination of temperature and economic
    objectives, depending on the reward_mode.
    MPC controller with optional objective:
        - 'temperature' or 'combined' (incl. economic costs).
    ----------------------------------------
    This controller is based on a simplified building model and optimizes over
    a short horizon. This class implements a minimal MPC scheme over a short time
    horizon. In `predict`, a pyomo model is constructed, solved and the resulting
    first control value (action) is returned.
    """

    def __init__(
        self,
        dt=300,
        horizon=3,
        building_params=None,
        reward_mode="temperature",
        temperature_weight=1.0,
        economic_weight=1.0,
        price_max=1.0,
        action_reg=0.0,  # 0.01 in the published evaluation runs
        action_smoothing=0.0,  # 0.05 in the published evaluation runs
        dt_scale=1e-3,
    ):
        """
        Initialize the MPC parameters and the optimization preferences.
        Args:
            dt (int, optional): Discretization timestep in seconds. Defaults to 300.
            horizon (int, optional): Number of prediction steps in the optimization.
                Defaults to 3.
            building_params (dict, optional): Contains building constants
                (mC, K, Q_HP_Max, T_set). If None, default values are used.
            reward_mode (str, optional): Either "temperature" or "combined".
                Determines the form of the objective function. Defaults to "temperature".
            temperature_weight (float, optional): Weight for the temperature objective
                in the combined mode. Defaults to 1.0.
            economic_weight (float, optional): Weight for the economic objective
                in the combined mode. Defaults to 1.0.
            price_max (float, optional): Normalization constant for the price, matching
                max_price in the environment reward. Defaults to 1.0.
            action_reg (float, optional): Weight of the optional action regularizer
                (action_reg / horizon) * a^2. Purely numerical, not part of the
                scored reward; it pins the weakly determined last actions of the
                horizon. Defaults to 0.0 (plain objective); the published
                evaluation runs used 0.01.
            action_smoothing (float, optional): Weight of the optional smoothing
                term action_smoothing * (a_k - a_{k-1})^2 within the horizon.
                Purely numerical, not part of the scored reward. Defaults to 0.0
                (plain objective); the published evaluation runs used 0.05.
            dt_scale (float, optional): Scale factor for the internal dynamics
                model, must match the environment's own dt_scale (see
                BaseBuildingGym/Building) or the MPC's predictions will be
                systematically wrong. Defaults to 1e-3 (toy-building
                calibration); pass 1.0 when mC/K/Q_HP_Max are true SI values.
        """
        self.dt = dt
        self.horizon = horizon
        self.reward_mode = reward_mode
        self.temperature_weight = temperature_weight
        self.economic_weight = economic_weight
        self.price_max = float(price_max) if price_max else 1.0
        self.action_reg = float(action_reg)
        self.action_smoothing = float(action_smoothing)
        self.dt_scale = float(dt_scale)

        # Default building parameters if none are provided
        if building_params is None:
            # Default values, if not passed
            self.building_params = {"mC": 300, "K": 20, "Q_HP_Max": 1500, "T_set": 25.0}
        else:
            self.building_params = building_params

    def predict(self, obs, deterministic=True, **kwargs):
        """
        Builds a Pyomo optimization model, solves it, and returns the first optimal action.

        Args:
            obs (np.array): Observations from the environment.
                Typically, obs[0] = T_in - T_set, so indoor temperature can be recovered.
            deterministic (bool, optional): Ignored in this simple MPC,
                but kept for interface consistency. Defaults to True.
            **kwargs:
                T_out_pred (list or np.array): Outdoor temperature forecast of length horizon+1.
                price_pred (list or np.array): Energy price forecast of length horizon.
                Additional parameters could be passed here if needed.

        Returns:
            tuple:
                - action (np.array): The first control action in the interval [-1, 1].
                - None: Placeholder for compatibility with controllers returning (action, state).
        """
        # 1) Extract relevant state variables
        # If obs[0] = (T_in - T_set) => T_in = obs[0] + T_set
        # Planning horizon in number of steps
        H = self.horizon
        T_set_list = list(
            kwargs.get(
                "T_set_pred", [self.building_params["T_set"]] * (self.horizon + 1)
            )
        )

        if len(T_set_list) < self.horizon + 1:
            last_value = (
                T_set_list[-1] if len(T_set_list) > 0 else self.building_params["T_set"]
            )
            T_set_list = list(T_set_list) + [last_value] * (
                self.horizon + 1 - len(T_set_list)
            )
        else:
            T_set_list = list(T_set_list[: self.horizon + 1])

        T_in_current = float(obs[0] + T_set_list[0])

        # Retrieve or default to dummy predictions
        T_out_list = kwargs.get("T_out_pred", [30.0] * (self.horizon + 1))

        # Price forecast over the horizon. Without a forecast the economic term
        # is inactive, a zero price is used instead of a fabricated default.
        price_pred = kwargs.get("price_pred", None)
        if price_pred is None:
            price_list = [0.0] * H
        else:
            price_list = [float(p) for p in price_pred]
            if len(price_list) < H:
                last_value = price_list[-1] if len(price_list) > 0 else 0.0
                price_list = price_list + [last_value] * (H - len(price_list))
            else:
                price_list = price_list[:H]

        mC = self.building_params["mC"]
        K = self.building_params["K"]
        Q_HP_Max = self.building_params["Q_HP_Max"]
        dt = self.dt

        # Safely retrieve T_out_pred
        T_out_pred = kwargs.get("T_out_pred", None)
        print(f"[DEBUG] Using T_set: {T_set_list}")

        # 2) Set T_out_list
        if T_out_pred is None:
            # Use dummy values if no prediction was given
            T_out_list = [30.0] * (self.horizon + 1)
            print(f"[DEBUG] Using dummy T_out_pred: {T_out_list}")

        else:
            T_out_pred = list(T_out_pred)  # Ensure it's a list
            if len(T_out_pred) < self.horizon + 1:
                print(
                    f"[DEBUG] T_out_pred too short (got {len(T_out_pred)}), extending with last value."
                )
                last_value = T_out_pred[-1] if len(T_out_pred) > 0 else 30.0
                T_out_list = T_out_pred + [last_value] * (
                    self.horizon + 1 - len(T_out_pred)
                )
            else:
                T_out_list = T_out_pred[: self.horizon + 1]
            print(f"[DEBUG] Using provided T_out_pred: {T_out_list}")

        # 3) Define Pyomo model
        model = pyo.ConcreteModel()
        # Control and state variables
        model.t = pyo.RangeSet(0, H - 1)
        # State variables: T_in[t] => Temperatur Indoor
        model.T_in = pyo.Var(range(self.horizon + 1), domain=pyo.Reals)
        # Control variables: action[t] in [-1,1]
        model.action = pyo.Var(model.t, domain=pyo.Reals, bounds=(-1, 1))
        # Initial condition
        model.T_in[0].fix(T_in_current)
        scale = self.dt_scale  # Must match the environment's Building.dt_scale

        # 3) Define dynamic equations

        def dyn_rule(m, i):
            # first use T_out_list[i] here
            if i >= len(T_out_list):
                # Do not use T_out_list[i] at all
                return pyo.Constraint.Skip
            if i == H:
                return pyo.Constraint.Skip
            return m.T_in[i + 1] == m.T_in[i] - scale * (dt / mC) * (
                K * (m.T_in[i] - T_out_list[i]) + m.action[i] * Q_HP_Max
            )

        model.dynamics = pyo.Constraint(model.t, rule=dyn_rule)

        # 4) Define objective function
        def objective_rule(m):
            """
            Objective function that can be temperature-only or combined with
            an economic term, depending on self.reward_mode. The two
            regularizers are optional numerical smoothers (zero weight
            disables them) and not part of the scored reward.
            """
            temp_penalty = sum((m.T_in[i] - T_set_list[i]) ** 2 for i in m.t)
            smooth_penalty = sum(
                self.action_smoothing * (m.action[i] - m.action[i - 1]) ** 2
                for i in range(1, H)
            )
            # Scale weight per time step => Horizon length does not influence the optimum
            reg_penalty = sum((self.action_reg / H) * m.action[i] ** 2 for i in m.t)

            if self.reward_mode != "combined":
                return temp_penalty + reg_penalty + smooth_penalty

            # Economic term, mirroring reward_economic_norm of the environment:
            # price * |action| / max_price, both terms weighted as in the reward.
            econ_penalty = sum(
                price_list[i]
                * pyo.sqrt(m.action[i] ** 2 + PRICE_ABS_EPS)
                / self.price_max
                for i in m.t
            )
            return (
                self.temperature_weight * temp_penalty
                + self.economic_weight * econ_penalty
                + reg_penalty
                + smooth_penalty
            )

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve the optimization model
        # IPOPT_EXECUTABLE lets HPC deployments point at a non-PATH install
        # (e.g. the KIT cluster's ~/.local/bin/ipopt); by default we let
        # Pyomo resolve "ipopt" from PATH so this works on any machine.
        ipopt_executable = os.environ.get("IPOPT_EXECUTABLE", "ipopt")
        solver = pyo.SolverFactory("ipopt", executable=ipopt_executable)
        solver.solve(model, tee=False)

        # Extract the first optimal action
        action_first = pyo.value(model.action[0])

        # Clipping to [-1, +1] if numeric limits are exceeded
        action_clipped = np.clip(action_first, -1.0, 1.0)

        return np.array([action_clipped]), None
