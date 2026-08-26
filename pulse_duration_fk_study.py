"""
Pulse-duration study.

Question: as the noise spectrum's knee frequency f_k sweeps through the
pulse's own bandwidth (~1/tau), how does the noise picked up DURING the
finite-duration drive itself change?

This deliberately isolates the in-pulse-only noise-coupling mechanism
from CPMG free-evolution decoupling, by using
N_pulse_sequence.FastFrequencyNoisePulseSequence rather than
NoisyFreeEvolutionPulseSequence. That separation is the point: results
here reflect gate-level dephasing during the drive, not dynamical
decoupling.
"""

import os
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from N_pulse_sequence import run_single_shot

OUTPUT_DIR = (
    "/Users/will/Library/Mobile Documents/com~apple~CloudDocs/"
    "Summer 26 internship/qitcat-main/Will_experiment/Will_data"
)


def make_run_dir():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"pulse_duration_fk_study_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Sweep parameters

N_RUNS = 200

PULSE_DURATIONS = [2e-6, 10e-6, 50e-6, 200e-6]  # s
F_K_LIST_HZ = [5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3]  # spans 1/tau for all tau above

N_PI_FIXED = 2  # fixed CPMG sequence 
T_TOTAL = 20e-3

ILW = 864
NOISE_FRACTION = 0.05


def run_sweep(run_dir):
    mean_grid = np.zeros((len(PULSE_DURATIONS), len(F_K_LIST_HZ)))
    std_grid = np.zeros_like(mean_grid)
    sigma_phi_grid = np.zeros_like(mean_grid)

    metadata = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        study="pulse_duration_vs_fk",
        noise_coupling="in-pulse-only (free_evolution_noise=False)",
        N_RUNS=N_RUNS,
        PULSE_DURATIONS=PULSE_DURATIONS,
        F_K_LIST_HZ=F_K_LIST_HZ,
        N_PI_FIXED=N_PI_FIXED,
        T_TOTAL=T_TOTAL,
        ILW=ILW,
        NOISE_FRACTION=NOISE_FRACTION,
    )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            ["pulse_duration_s", "f_k_hz", "n_runs", "mean_Pe", "std_Pe",
             "sigma_phi_rad", "raw_data_file"]
        )

        for i, tau in enumerate(PULSE_DURATIONS):
            for j, f_k in enumerate(F_K_LIST_HZ):

                noise_kwargs = dict(ILW=ILW, f_k=f_k, noise_fraction=NOISE_FRACTION)

                probs = np.empty(N_RUNS)
                for k in range(N_RUNS):
                    probs[k] = run_single_shot(
                        n_pi_pulses=N_PI_FIXED,
                        pulse_duration=tau,
                        T_total=T_TOTAL,
                        noise_kwargs=noise_kwargs,
                        free_evolution_noise=False,  # isolate in-pulse-only coupling
                    )

                mean_grid[i, j] = probs.mean()
                std_grid[i, j] = probs.std(ddof=1)
                sigma_phi_grid[i, j] = 2.0 * std_grid[i, j]

                raw_filename = f"probs_tau{tau*1e6:.1f}us_fk{f_k/1e3:.0f}kHz.npy"
                np.save(os.path.join(run_dir, raw_filename), probs)

                writer.writerow(
                    [tau, f_k, N_RUNS, mean_grid[i, j], std_grid[i, j],
                     sigma_phi_grid[i, j], raw_filename]
                )
                summary_file.flush()

                print(
                    f"tau={tau*1e6:6.1f}us  f_k={f_k/1e3:7.1f}kHz  "
                    f"sigma_phi~{sigma_phi_grid[i, j]:.3e} rad"
                )

    print(f"\nSaved pulse-duration study data to: {run_dir}")
    return mean_grid, std_grid, sigma_phi_grid


def plot_results(sigma_phi_grid, run_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, tau in enumerate(PULSE_DURATIONS):
        ax.loglog(
            F_K_LIST_HZ,
            sigma_phi_grid[i, :],
            marker="o",
            label=f"tau = {tau*1e6:.0f} us (1/tau = {1/tau/1e3:.0f} kHz)",
        )

    ax.set_xlabel(r"Noise knee frequency $f_k$ (Hz)")
    ax.set_ylabel(r"Inferred phase noise $\sigma_\varphi$ (rad)")
    ax.set_title(
        "In-pulse-only noise vs $f_k$, by pulse duration\n"
        f"(N_pi={N_PI_FIXED} fixed, free evolution excluded)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pulse_duration_fk_study.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    run_dir = make_run_dir()
    mean_grid, std_grid, sigma_phi_grid = run_sweep(run_dir)
    plot_results(sigma_phi_grid, run_dir)
