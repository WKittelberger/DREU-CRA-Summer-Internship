"""
N-pulse study.

Question: is the CPMG N_pi-dependence sensitive to where the noise
spectrum's knee f_k sits, relative to the sequence's own timescales? 
The filter-function plot showed each N has a widening low-frequency 
rejection band. this study sweeps f_k through that region directly to see if N-dependence appears.

Run at every (N, f_k) grid point, exactly the same way run_N_pulse_sequence.py 
did for the original flat sweep, just now with f_k swept into the sensitive 
band instead of fixed at 1.16 MHz. This is the trustworthy, quantitative result.
"""

import os
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from N_pulse_sequence import run_single_shot, NoisyFreeEvolutionPulseSequence
from filter_quadrature import (
    compute_filter_function_quadrature,
    measure_empirical_psd,
    predict_sigma_phi_squared,
    resample_response_to_psd_grid,
)

OUTPUT_DIR = (
    "/Users/will/Library/Mobile Documents/com~apple~CloudDocs/"
    "Summer 26 internship/qitcat-main/Will_experiment/Will_data"
)


def make_run_dir():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"n_pulses_fk_study_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# parameters

N_PI_PULSES_LIST = [0, 1, 2, 4, 8, 16]
F_K_LIST_HZ = [50.0, 100.0, 200.0, 500.0, 1000.0]

PULSE_DURATION = 10e-6
T_TOTAL = 20e-3

ILW = 864
NOISE_FRACTION = 0.05

N_RUNS = 1000 

# Frequency grid for the response curve
RESPONSE_FREQS_HZ = np.logspace(np.log10(0.5), np.log10(2.5e4), 50)

# fast qualitative landscape

def run_qualitative_landscape(run_dir):
    """
    Compute the response curve once per N, then cheaply evaluate the
    Parseval overlap for every f_k. Saves a landscape CSV and plot.
    Returns the predicted sigma_phi grid.
    """
    print("Stage 1: fast qualitative landscape (no Monte Carlo)...")

    responses = {}
    for n_pi in N_PI_PULSES_LIST:
        t0 = time.time()
        responses[n_pi] = compute_filter_function_quadrature(
            NoisyFreeEvolutionPulseSequence,
            n_pi,
            PULSE_DURATION,
            T_TOTAL,
            RESPONSE_FREQS_HZ,
            amplitude_hz=1.0,
        )
        print(f"  N={n_pi}: response curve computed in {time.time()-t0:.1f}s")

    landscape = np.zeros((len(N_PI_PULSES_LIST), len(F_K_LIST_HZ)))
    for j, f_k in enumerate(F_K_LIST_HZ):
        freqs_psd, psd = measure_empirical_psd(ILW, f_k, NOISE_FRACTION)
        for i, n_pi in enumerate(N_PI_PULSES_LIST):
            response_on_grid = resample_response_to_psd_grid(
                RESPONSE_FREQS_HZ, responses[n_pi], freqs_psd
            )
            landscape[i, j] = np.sqrt(
                predict_sigma_phi_squared(freqs_psd, response_on_grid, psd)
            )

    # Save landscape (clearly labeled as uncalibrated/relative)
    with open(os.path.join(run_dir, "qualitative_landscape.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_pi_pulses"] + [f"fk_{fk}Hz_UNCALIBRATED" for fk in F_K_LIST_HZ])
        for i, n_pi in enumerate(N_PI_PULSES_LIST):
            writer.writerow([n_pi] + list(landscape[i, :]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, f_k in enumerate(F_K_LIST_HZ):
        ax.plot(N_PI_PULSES_LIST, landscape[:, j], marker="o", label=f"f_k = {f_k:.0f} Hz")
    ax.set_xlabel("Number of pi pulses (N)")
    ax.set_ylabel(r"Predicted $\sigma_\varphi$ (UNCALIBRATED units)")
    ax.set_title("Stage 1: fast qualitative landscape (relative shape only)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "qualitative_landscape.png"), dpi=150)
    plt.close(fig)

    print("Stage 1 done.\n")
    return landscape


# full Monte Carlo confirmation

def run_monte_carlo_confirmation(run_dir):
    print("Stage 2: full Monte Carlo confirmation...")

    mean_grid = np.zeros((len(N_PI_PULSES_LIST), len(F_K_LIST_HZ)))
    std_grid = np.zeros_like(mean_grid)
    sigma_phi_grid = np.zeros_like(mean_grid)

    metadata = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        study="n_pulses_vs_fk",
        noise_coupling="full (free evolution + pulses, free_evolution_noise=True)",
        N_RUNS=N_RUNS,
        N_PI_PULSES_LIST=N_PI_PULSES_LIST,
        F_K_LIST_HZ=F_K_LIST_HZ,
        PULSE_DURATION=PULSE_DURATION,
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
            ["n_pi_pulses", "f_k_hz", "n_runs", "mean_Pe", "std_Pe",
             "sigma_phi_rad", "raw_data_file"]
        )

        for i, n_pi in enumerate(N_PI_PULSES_LIST):
            for j, f_k in enumerate(F_K_LIST_HZ):

                noise_kwargs = dict(ILW=ILW, f_k=f_k, noise_fraction=NOISE_FRACTION)

                probs = np.empty(N_RUNS)
                for k in range(N_RUNS):
                    probs[k] = run_single_shot(
                        n_pi_pulses=n_pi,
                        pulse_duration=PULSE_DURATION,
                        T_total=T_TOTAL,
                        noise_kwargs=noise_kwargs,
                        free_evolution_noise=True,
                    )

                mean_grid[i, j] = probs.mean()
                std_grid[i, j] = probs.std(ddof=1)
                sigma_phi_grid[i, j] = 2.0 * std_grid[i, j]

                raw_filename = f"probs_Npi{n_pi}_fk{f_k:.0f}Hz.npy"
                np.save(os.path.join(run_dir, raw_filename), probs)

                writer.writerow(
                    [n_pi, f_k, N_RUNS, mean_grid[i, j], std_grid[i, j],
                     sigma_phi_grid[i, j], raw_filename]
                )
                summary_file.flush()

                print(
                    f"N={n_pi:3d}  f_k={f_k:7.1f}Hz  "
                    f"sigma_phi~{sigma_phi_grid[i, j]:.3e} rad"
                )

    print(f"\nSaved N-pulse study data to: {run_dir}")
    return mean_grid, std_grid, sigma_phi_grid


def plot_mc_results(sigma_phi_grid, run_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, f_k in enumerate(F_K_LIST_HZ):
        ax.plot(N_PI_PULSES_LIST, sigma_phi_grid[:, j], marker="o", label=f"f_k = {f_k:.0f} Hz")
    ax.set_xlabel("Number of pi pulses (N)")
    ax.set_ylabel(r"Inferred phase noise $\sigma_\varphi$ (rad)")
    ax.set_title(
        f"Stage 2: Monte Carlo confirmation "
        f"(T_total={T_TOTAL*1e3:.0f}ms, tau={PULSE_DURATION*1e6:.0f}us)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "n_pulses_fk_mc_confirmation.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    run_dir = make_run_dir()
    run_qualitative_landscape(run_dir)
    mean_grid, std_grid, sigma_phi_grid = run_monte_carlo_confirmation(run_dir)
    plot_mc_results(sigma_phi_grid, run_dir)
