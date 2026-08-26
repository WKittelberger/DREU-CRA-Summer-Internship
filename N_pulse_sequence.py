"""
CPMG-style pulse-sequence construction on top of FrequencyNoisePulseSequence.

A CPMG (Carr-Purcell-Meiboom-Gill) interferometer sequence is:

    pi/2 -- tau -- pi -- tau -- pi -- ... -- pi -- tau -- pi/2

with n_pi_pulses + 1 equal free-evolution segments.

Two sweep knobs are supported, matching the two questions being asked
of the noise model:

    n_pi_pulses    -- how many pi-pulses are inserted
    pulse_duration -- finite duration of every pulse (pi/2 and pi alike,
                       consistent with FrequencyNoisePulseSequence's
                       single-pulse_duration convention)

By default the total free-precession time T_total is held fixed as 
n_pi_pulses is varied, so that increasing N changes the sequence's noise 
"filter function" rather than simply extending the total experiment time. 
This is the standard way dynamical decoupling is characterized. 
"""

import numpy as np

from pulse_sequence_frequency import FrequencyNoisePulseSequence

# numpy >= 2.0 renamed trapz -> trapezoid; keep this working either way.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class FastFrequencyNoisePulseSequence(FrequencyNoisePulseSequence):
    """
    Performance variant of FrequencyNoisePulseSequence.

    The propagator for a 2x2 Hermitian Hamiltonian has a closed analytic form:
        H = 0.5 * [Delta * sigma_z + Omega*cos(phase) * sigma_x
                    + Omega*sin(phase) * sigma_y]

        Omega_eff = sqrt(Omega^2 + Delta^2)
        theta_eff = Omega_eff * dt
        n_hat = (Omega*cos(phase), Omega*sin(phase), Delta) / Omega_eff

        U = cos(theta_eff/2) * I - i*sin(theta_eff/2) * (n_hat . sigma)

    This is mathematically identical to `expm(-1j * H * dt)` in the
    base class (validated in test_N_pulse_sequence.py) but 10-50x faster,
    since it replaces a general-purpose matrix-exponential algorithm
    with a handful of trig calls. Only `apply_noisy_pulse` is
    overridden; noise generation, free evolution, and run() are
    inherited unchanged.
    """

    def apply_noisy_pulse(self, state, pulse_area, t_start, phase=0.0):
        tau = self.pulse_duration
        Omega = pulse_area / tau

        n_steps = max(1, int(np.ceil(tau / self.pulse_dt)))
        dt = tau / n_steps

        c1, c2 = complex(state[0]), complex(state[1])

        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        for i in range(n_steps):
            t = t_start + (i + 0.5) * dt
            delta_nu = self.get_frequency_noise(t)
            Delta = 2.0 * np.pi * delta_nu

            Omega_eff = np.hypot(Omega, Delta)
            if Omega_eff == 0.0:
                continue

            theta = Omega_eff * dt
            cos_h = np.cos(0.5 * theta)
            sin_h = np.sin(0.5 * theta)

            nx = Omega * cos_p / Omega_eff
            ny = Omega * sin_p / Omega_eff
            nz = Delta / Omega_eff

            # U = cos_h*I - i*sin_h*(nx*sx + ny*sy + nz*sz)
            u00 = cos_h - 1j * sin_h * nz
            u01 = -1j * sin_h * (nx - 1j * ny)
            u10 = -1j * sin_h * (nx + 1j * ny)
            u11 = cos_h + 1j * sin_h * nz

            c1, c2 = (u00 * c1 + u01 * c2), (u10 * c1 + u11 * c2)

        return np.array([c1, c2], dtype=complex)


class NoisyFreeEvolutionPulseSequence(FastFrequencyNoisePulseSequence):
    """
    Extends the pulse sequence so laser frequency noise also accumulates
    phase during free-evolution segments, via

        H_free(t) = (Delta(t) / 2) * sigma_z,   Delta(t) = 2*pi*delta_nu(t)

    Because sigma_z at different times always commutes with itself, the
    exact propagator over a free segment [t_start, t_start + duration] is

        U = diag( exp(-i*phi/2), exp(+i*phi/2) ),
        phi = integral_{t_start}^{t_start+duration} Delta(t) dt

    so no matrix exponential is needed here -- phi is
    obtained by numerically integrating the already-generated noise
    realization.

    This is what lets CPMG pi-pulses actually refocus phase accumulated
    during free evolution: a pi-pulse conjugates sigma_z -> -sigma_z, so
    the sign of phase accumulated after a pi-pulse is automatically
    flipped relative to before it.
    The toggling-frame filter function y(t) = +/-1 emerges from this
    directly.

    Only `free_evolution` is overridden; noisy finite-duration pulses
    are inherited unchanged from FastFrequencyNoisePulseSequence.
    """

    def free_evolution(self, state, duration, t_start=None):
        if t_start is None:
            raise ValueError(
                "t_start is required for noise-aware free evolution."
            )

        if self.noise_t is None:
            raise RuntimeError("Noise has not been generated. Call run().")

        t_end = t_start + duration

        i0 = np.searchsorted(self.noise_t, t_start, side="left")
        i1 = np.searchsorted(self.noise_t, t_end, side="right")

        if i1 - i0 < 2:
            delta_nu_avg = self.get_frequency_noise(0.5 * (t_start + t_end))
            phi = 2.0 * np.pi * delta_nu_avg * duration
        else:
            t_seg = self.noise_t[i0:i1]
            nu_seg = self.noise_hz[i0:i1]
            phi = 2.0 * np.pi * _trapz(nu_seg, t_seg)

        c1, c2 = complex(state[0]), complex(state[1])
        half = 0.5 * phi

        c1_new = c1 * np.exp(-1j * half)
        c2_new = c2 * np.exp(1j * half)

        return np.array([c1_new, c2_new], dtype=complex)


def compute_toggling_frame_phase(
    noise_t,
    noise_hz,
    pulse_areas,
    pulse_duration,
    free_evolution_time,
):
    """
    Independent reference calculation of the accumulated phase in the
    idealized-pi-pulse limit:

        phi = integral_0^{T_total} y(t) * Delta(t) dt

    where y(t) = +1 before the first pi pulse, flips sign at every pi
    pulse, and pi/2 pulses do not flip it (they are the
    beamsplitter/mirror pulses that open and close the interferometer,
    not part of the echo train).

    This ignores finite pulse duration entirely, so it should only be compared against
    NoisyFreeEvolutionPulseSequence results in the tau_pulse -> 0 limit.
    It exists purely to validate that the full finite-duration
    simulation reduces to the expected analytic dynamical-decoupling
    behavior.

    Parameters
    ----------
    noise_t, noise_hz :
        The realization returned by generate_noise().
    pulse_areas
    pulse_duration
    free_evolution_time :
        Length len(pulse_areas) - 1.

    Returns
    -------
    phi :
        Accumulated phase in the idealized toggling-frame picture.
    """
    pulse_areas = np.asarray(pulse_areas, dtype=float)
    free_evolution_time = np.asarray(free_evolution_time, dtype=float)

    # Pulse edge times, ignoring pulse duration (pulses idealized as instantaneous
    # and placed at the midpoint of where the finite-duration pulse would sit).
    t = 0.0
    pulse_times = []
    segment_bounds = []  # (t_start, t_end, y_sign) for each free segment

    y = +1.0
    for i in range(len(pulse_areas)):
        pulse_times.append(t)
        t += pulse_duration
        if i < len(free_evolution_time):
            t_seg_start = t
            t_seg_end = t + free_evolution_time[i]
            segment_bounds.append((t_seg_start, t_seg_end, y))
            t = t_seg_end
            # pi pulses (area ~ pi) flip the toggling sign; pi/2 pulses do not
            if np.isclose(pulse_areas[i + 1], np.pi):
                y = -y

    phi = 0.0
    for t_start, t_end, y_sign in segment_bounds:
        i0 = np.searchsorted(noise_t, t_start, side="left")
        i1 = np.searchsorted(noise_t, t_end, side="right")
        if i1 - i0 < 2:
            continue
        seg_phi = 2.0 * np.pi * _trapz(noise_hz[i0:i1], noise_t[i0:i1])
        phi += y_sign * seg_phi

    return phi


def build_n_pulse_sequence(
    n_pi_pulses,
    pulse_duration,
    T_total=None,
    segment_spacing=None,
    mode="fixed_total_time",
    readout_phase=np.pi / 2,
):
    """
    Uses standard CPMG timing: pulses are placed at
    t_j = (j - 1/2) * tau for j = 1..N, where tau is the inter-pulse
    spacing. This means the FIRST and LAST free-evolution segments are
    tau/2, and the N-1 segments between consecutive pi-pulses are tau.

    Parameters
    ----------
    n_pi_pulses :
        Number of pi-pulses between the two pi/2 pulses. 0 recovers a
        plain Ramsey sequence [pi/2, pi/2].
    pulse_duration : float
        Duration of every pulse.
    T_total :
        Total free precession time, i.e. the sum of all dark
        segments (unchanged meaning: tau/2 + (N-1)*tau + tau/2 = N*tau
        = T_total still holds under the corrected timing).
    segment_spacing :
        Interpulse spacing tau -- the duration of each MIDDLE
        segment; the two end segments will be tau/2.
    mode : {'fixed_total_time', 'fixed_spacing'}.
    readout_phase :
        Phase offset (rad) applied to the final pi/2 pulse only.
        Default pi/2 places the sequence at mid-fringe, where P_e
        responds linearly to accumulated phase noise. Use 0.0 to
        recover the dark-port operating point.

    Returns
    -------
    pulse_areas : 
    free_evolution_time : 
        Length n_pi_pulses + 1. [tau/2, tau, tau, ..., tau, tau/2] for
        n_pi_pulses >= 1; [T_total] for n_pi_pulses == 0.
    pulse_phase_offsets : 
        Length n_pi_pulses + 2, matching pulse_areas.
    """
    if n_pi_pulses < 0:
        raise ValueError("n_pi_pulses must be >= 0.")

    if n_pi_pulses == 0:
        if mode == "fixed_total_time":
            if T_total is None:
                raise ValueError("T_total is required for mode='fixed_total_time'.")
            free_evolution_time = [T_total]
        elif mode == "fixed_spacing":
            if segment_spacing is None:
                raise ValueError("segment_spacing is required for mode='fixed_spacing'.")
            free_evolution_time = [segment_spacing]
        else:
            raise ValueError("mode must be 'fixed_total_time' or 'fixed_spacing'.")
    else:
        if mode == "fixed_total_time":
            if T_total is None:
                raise ValueError("T_total is required for mode='fixed_total_time'.")
            tau = T_total / n_pi_pulses
        elif mode == "fixed_spacing":
            if segment_spacing is None:
                raise ValueError("segment_spacing is required for mode='fixed_spacing'.")
            tau = segment_spacing
        else:
            raise ValueError("mode must be 'fixed_total_time' or 'fixed_spacing'.")

        free_evolution_time = [tau / 2.0] + [tau] * (n_pi_pulses - 1) + [tau / 2.0]

    pulse_areas = [np.pi / 2] + [np.pi] * n_pi_pulses + [np.pi / 2]

    pulse_phase_offsets = [0.0] * len(pulse_areas)
    pulse_phase_offsets[-1] = readout_phase

    return pulse_areas, free_evolution_time, pulse_phase_offsets


def run_single_shot(
    n_pi_pulses,
    pulse_duration,
    T_total,
    noise_kwargs,
    noise_dt=1e-7,
    pulse_dt=1e-7,
    readout_phase=np.pi / 2,
    mode="fixed_total_time",
    segment_spacing=None,
    fast=True,
    free_evolution_noise=True,
):
    """
    Parameters
    ----------
    fast :
        If True (default), use a closed-form propagator for the pulses
        instead of expm. Both give numerically identical results; 
        fast is only about speed.
    free_evolution_noise :
        If True (default), laser frequency noise accumulates phase
        during free-evolution segments too, enabling genuine CPMG echo/refocusing behavior. 
        If False, free evolution is a no-op (original duty-cycle-only model).
    Returns
    ----------
    Probability
    """
    from qitcat.modules.noise import FrequencyNoise

    pulse_areas, free_evolution_time, pulse_phase_offsets = build_n_pulse_sequence(
        n_pi_pulses=n_pi_pulses,
        pulse_duration=pulse_duration,
        T_total=T_total,
        segment_spacing=segment_spacing,
        mode=mode,
        readout_phase=readout_phase,
    )

    noise = FrequencyNoise(**noise_kwargs)

    if free_evolution_noise:
        seq_cls = NoisyFreeEvolutionPulseSequence
    elif fast:
        seq_cls = FastFrequencyNoisePulseSequence
    else:
        seq_cls = FrequencyNoisePulseSequence

    sequence = seq_cls(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_evolution_time,
        noise_model=noise,
        noise_dt=noise_dt,
        pulse_dt=pulse_dt,
    )

    _, probability = sequence.run(pulse_phase_offsets=pulse_phase_offsets)
    return probability