"""
Compute and plot the filter function of CPMG sequences for several values of N.
"""

import numpy as np
import matplotlib.pyplot as plt

from filter_function import compute_filter_function

# Sweep parameters

N_PI_PULSES_LIST = [0, 1, 2, 4, 8]
PULSE_DURATION = 10e-6  # (s)
T_TOTAL = 20e-3  # (s) fixed across all curves

# The filter function's lobe structure has scale ~1/(2*T_total) = 25 Hz for T_total=20ms. 
_freqs_low = np.logspace(0, 2, 25)       # 1-100 Hz
_freqs_high = np.linspace(100, 2000, 400)  # 100 Hz - 2 kHz
FREQS_HZ = np.unique(np.concatenate([_freqs_low, _freqs_high]))

AMPLITUDE_HZ = 1.0  # tone amplitude; stays in the linear regime 


def run_and_plot():
    fig, ax = plt.subplots(figsize=(8, 6))

    for n_pi in N_PI_PULSES_LIST:
        response = compute_filter_function(
            n_pi_pulses=n_pi,
            pulse_duration=PULSE_DURATION,
            T_total=T_TOTAL,
            freqs_hz=FREQS_HZ,
            amplitude_hz=AMPLITUDE_HZ,
        )
        label = "N=0 (Ramsey)" if n_pi == 0 else f"N={n_pi}"
        ax.loglog(FREQS_HZ, response, label=label)
        print(f"N={n_pi} done")

    ax.set_xlabel("Tone frequency (Hz)")
    ax.set_ylabel(r"Response $|\phi(f)|$ / tone amplitude (rad/Hz)")
    ax.set_title(
        f"CPMG filter function (T_total = {T_TOTAL*1e3:.0f} ms, "
        f"pulse duration = {PULSE_DURATION*1e6:.0f} us)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("filter_function.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_and_plot()