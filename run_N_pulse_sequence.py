"""
Sweep laser-frequency-noise statistics over number of pi-pulses 
inserted into a CPMG sequence, and finite pulse duration

Laser frequency noise now accumulates phase during BOTH the finite
pulses AND the free-evolution segments (H_free = Delta(t)/2 *
sigma_z), so pi-pulses refocus phase picked up during free evolution

The linear approximation for σ_φ only holds while δφ stays small 
(roughly δφ ≲ 0.5 rad before the sin curvature starts mattering). 

Total free-precession time T_total is held across the N sweep
(mode='fixed_total_time'), so varying N changes the sequence's noise
filter function rather than the total experiment duration.

Note: qitcat's FrequencyNoise forces the DC-bin phase to zero (a
standard real-FFT trick), which makes each realization's mean value a
near-deterministic small positive bias rather than a genuine zero-mean
random variable. This is refocused away by any N>=1 CPMG sequence but
dominates a plain Ramsey (N=0) measurement.
"""

import os
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from N_pulse_sequence import run_single_shot  # fast=True by default

OUTPUT_DIR = (
    "/Users/will/Library/Mobile Documents/com~apple~CloudDocs/"
    "Summer 26 internship/qitcat-main/Will_experiment/Will_data"
)


def make_run_dir():
    """Create a fresh timestamped subfolder under OUTPUT_DIR for this
    sweep, so repeated runs don't overwrite each other's data."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"n_pulse_sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Sweep parameters

N_RUNS = 200  

N_PI_PULSES_LIST = [0, 1, 2, 4, 8, 16]
PULSE_DURATIONS = [2e-6, 10e-6, 50e-6, 200e-6]  

T_TOTAL = 20e-3  # total free-precession time

NOISE_KWARGS = dict(ILW=864, f_k=1.16e6, noise_fraction=0.05)

READOUT_PHASE = np.pi / 2  # mid-fringe

# Run the sweep

def run_sweep(run_dir):
  
    mean_grid = np.zeros((len(PULSE_DURATIONS), len(N_PI_PULSES_LIST)))
    var_grid = np.zeros_like(mean_grid)
    std_grid = np.zeros_like(mean_grid)
    sigma_phi_grid = np.zeros_like(mean_grid)  # inferred phase-noise std

    metadata = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        N_RUNS=N_RUNS,
        N_PI_PULSES_LIST=N_PI_PULSES_LIST,
        PULSE_DURATIONS=PULSE_DURATIONS,
        T_TOTAL=T_TOTAL,
        NOISE_KWARGS=NOISE_KWARGS,
        READOUT_PHASE=READOUT_PHASE,
    )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                "pulse_duration_s",
                "n_pi_pulses",
                "n_runs",
                "mean_Pe",
                "var_Pe",
                "std_Pe",
                "sigma_phi_rad",
                "raw_data_file",
            ]
        )

        for i, tau in enumerate(PULSE_DURATIONS):
            for j, n_pi in enumerate(N_PI_PULSES_LIST):

                probs = np.empty(N_RUNS)
                for k in range(N_RUNS):
                    probs[k] = run_single_shot(
                        n_pi_pulses=n_pi,
                        pulse_duration=tau,
                        T_total=T_TOTAL,
                        noise_kwargs=NOISE_KWARGS,
                        readout_phase=READOUT_PHASE,
                    )

                mean_grid[i, j] = probs.mean()
                var_grid[i, j] = probs.var(ddof=1)
                std_grid[i, j] = probs.std(ddof=1)
                # Linear mid-fringe readout: P_e ~= 0.5 + 0.5*sin(delta_phi) ~= 0.5 + 0.5*delta_phi
                sigma_phi_grid[i, j] = 2.0 * std_grid[i, j]

                raw_filename = f"probs_tau{tau*1e6:.1f}us_Npi{n_pi}.npy"
                np.save(os.path.join(run_dir, raw_filename), probs)

                writer.writerow(
                    [
                        tau,
                        n_pi,
                        N_RUNS,
                        mean_grid[i, j],
                        var_grid[i, j],
                        std_grid[i, j],
                        sigma_phi_grid[i, j],
                        raw_filename,
                    ]
                )
                summary_file.flush()

                print(
                    f"tau={tau:.1e}s  N_pi={n_pi:3d}  "
                    f"mean={mean_grid[i, j]:.6f}  std={std_grid[i, j]:.3e}  "
                    f"sigma_phi~{sigma_phi_grid[i, j]:.3e} rad"
                )

    print(f"\nSaved sweep data to: {run_dir}")
    return mean_grid, var_grid, std_grid, sigma_phi_grid


def plot_results(std_grid, sigma_phi_grid, run_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Line plot
    ax = axes[0]
    for i, tau in enumerate(PULSE_DURATIONS):
        ax.plot(
            N_PI_PULSES_LIST,
            sigma_phi_grid[i, :],
            marker="o",
            label=f"tau = {tau*1e6:.0f} us",
        )
    ax.set_xlabel("Number of pi pulses (N)")
    ax.set_ylabel(r"Inferred phase noise $\sigma_\varphi$ (rad)")
    ax.set_title(f"Phase noise vs N (T_total = {T_TOTAL*1e3:.0f} ms fixed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heatmap
    ax = axes[1]
    im = ax.imshow(
        sigma_phi_grid,
        aspect="auto",
        origin="lower",
        extent=[
            min(N_PI_PULSES_LIST) - 0.5,
            max(N_PI_PULSES_LIST) + 0.5,
            0,
            len(PULSE_DURATIONS),
        ],
        cmap="viridis",
    )
    ax.set_xticks(N_PI_PULSES_LIST)
    ax.set_yticks(np.arange(len(PULSE_DURATIONS)) + 0.5)
    ax.set_yticklabels([f"{tau*1e6:.0f} us" for tau in PULSE_DURATIONS])
    ax.set_xlabel("Number of pi pulses (N)")
    ax.set_ylabel("Pulse duration")
    ax.set_title(r"$\sigma_\varphi$ (rad) heatmap")
    fig.colorbar(im, ax=ax, label=r"$\sigma_\varphi$ (rad)")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "n_pulse_noise_sweep.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    run_dir = make_run_dir()
    mean_grid, var_grid, std_grid, sigma_phi_grid = run_sweep(run_dir)
    plot_results(std_grid, sigma_phi_grid, run_dir)