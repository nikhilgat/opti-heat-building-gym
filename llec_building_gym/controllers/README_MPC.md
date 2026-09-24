# MPC Controller and Solver Installation on HPC (`README_MPC.md`)

## MPC controller

`mpc_controller.py` solves a constrained optimization problem over a finite horizon of `H` steps at every control step and applies the first action (receding horizon). The objective follows `reward_mode`:

| `reward_mode` | Objective | Effect | Code |
|---|---|---|---|
| `temperature` | $\min_{\{a_k\}} \; \sum_{k=0}^{H-1} \bigl(T_{\mathrm{in},k} - T_{\mathrm{set},k}\bigr)^2$ | • tracks the setpoint only | [`mpc_controller.py`, line 208](mpc_controller.py#L208) |
| `combined` | $\min_{\{a_k\}} \; \underbrace{w_{\mathrm{temp}} \sum_{k=0}^{H-1} \bigl(T_{\mathrm{in},k} - T_{\mathrm{set},k}\bigr)^2}_{\text{temperature term}} + \underbrace{w_{\mathrm{econ}} \sum_{k=0}^{H-1} \frac{\hat{p}_k \, \lvert a_k \rvert}{p_{\max}}}_{\text{economic term}}$ | • adds the economic term of the environment reward<br>• the controller minimizes the objective it is scored on | [`mpc_controller.py`, lines 212–223](mpc_controller.py#L212-L223) |

The absolute value $\lvert a_k \rvert$ in the economic term is implemented as $\sqrt{a_k^2 + 10^{-6}}$ ([`mpc_controller.py`, line 214](mpc_controller.py#L214), constant `PRICE_ABS_EPS`), so the objective stays differentiable at zero for IPOPT. Defaults are `w_temp = w_econ = 1.0` and `p_max = 1.0`, matching `max_price` in [`base_building_gym.py`, line 708](../envs/base_building_gym.py#L708).

Two optional numerical regularizers can be added to either mode via the constructor arguments `action_reg` and `action_smoothing`. They are not part of the scored reward.

| `action_reg` | `action_smoothing` | Objective | Effect |
|---|---|---|---|
| `0.0` | `0.0` | default, the plain objective above in both modes | • the controller optimizes exactly the terms it is scored on |
| $\neq 0$ | — | $+ \; \frac{\texttt{action\_reg}}{H} \sum_{k=0}^{H-1} a_k^2$ (scaled by $H$, so the total weight is horizon-independent) | • pins the weakly determined last actions of the horizon toward zero<br>• numerical well-posedness for IPOPT |
| — | $\neq 0$ | $+ \; \texttt{action\_smoothing} \cdot \sum_{k=1}^{H-1} \bigl(a_k - a_{k-1}\bigr)^2$ (starts at $k = 1$, no comparison against the previously applied action) | • damps chattering between consecutive actions<br>• smoother heat pump operation |
| `0.01` | `0.05` | $+ \; \frac{0.01}{H} \sum_{k=0}^{H-1} a_k^2 \; + \; 0.05 \cdot \sum_{k=1}^{H-1} \bigl(a_k - a_{k-1}\bigr)^2$ | • reproduces the published evaluation runs<br>• pass these values explicitly |

Note (2026-08-26): the economic term was previously commented out, so the MPC optimized comfort only, even under `reward_mode="combined"`. Enabling it shifts the C01 results by less than 0.4 % and leaves the controller ranking unchanged, since the comfort term outweighs the price term by an order of magnitude in the heat balance and the 60 min horizon rules out load shifting within constant hourly prices. The temperature mode is bit-identical to before. Reproduce with `sbatch slurm_script/slurm_eval_02_mpc.sh` ({MPC, Perfect MPC} x {T01, C01}, 10 episodes, seed 58, horizon 12).

## Solver installation

These guidelines describe the installation and configuration of the IPOPT solver in the HPC environment, including the necessary environment variables and the installation of additional solvers helpers.

### 1. IPOPT Installation

1. incompatibilities when linking the Intel MKL libraries Intel problems
If problems occur, often due to incompatibilities in the linkage of the Intel MKL libraries (`static linkage`,`libmkl_avx512.so.2`), cloning and setting up ThirdParty ASL can help. This solves certain dependencies for ASL that Ipopt requires in some configurations.

    ```bash
    git clone https://github.com/coin-or-tools/ThirdParty-ASL.git
    cd ThirdParty-ASL/
    ./get.ASL
    ./configure --prefix=${HOME}/.local
    make
    make install
    ```

2. Change to the parent directory, download Ipopt, unpack the archive, configure and install it.

    ```bash
    cd ..
    wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.14.4.tar.gz
    tar -xvzf Ipopt-3.14.4.tar.gz
    cd Ipopt-releases-3.14.4/
    ./configure --prefix=${HOME}/.local --with-lapack-lflags="-Wl,--no-as-needed -Wl,--start-group,${MKLROOT}/lib/intel64/libmkl_intel_lp64.a,${MKLROOT}/lib/intel64/libmkl_gnu_thread.a,${MKLROOT}/lib/intel64/libmkl_core.a,--end-group -lgomp -lpthread -lm -ldl"
    make
    make test
    make install
    ```

## 2.GLPK Installation

1. Step: Download the software
2. Step: Unpack archive
3. Step: Change to the directory
4. Step: Configuration
5. Step: Installation
6. Step: Set environment variables -- In order to be able to use the installed solver, the `PATH` and `LD_LIBRARY_PATH` environment variables must be adjusted.

    ```bash
    wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
    tar xzf glpk-5.0.tar.gz
    cd glpk-5.0/
    ./configure --prefix=$HOME/.local
    make install
    export PATH=$PATH:~/.local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib
    ```