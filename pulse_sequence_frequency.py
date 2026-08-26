"""
Custom pulse-sequence simulator for studying laser frequency noise.

This module is intentionally simpler than QitCats gravimeter model.

It models a two-level atom interacting with a sequence of Raman pulses
while being subjected only to time-dependent laser frequency noise.

    pi/2 -- T -- pi -- T -- pi/2

finite-duration pulses.

The laser frequency noise is generated once for the entire experiment,
so every pulse samples the same continuous noise realization.

No gravity, vibration, thermal velocity, phase noise, or other noise
sources are included.
"""

import numpy as np
from scipy.linalg import expm


class FrequencyNoisePulseSequence:
    """
    Simulate atom under a configurable pulse sequence
    with time-dependent laser frequency noise.

    Parameters
    ----------
    pulse_areas :
        Rotation angles for each pulse (rads).

    pulse_duration :
        Duration of every pulse in seconds.

    free_evolution_time :
        Time between pulses in seconds.

    noise_model :
        QitCat FrequencyNoise object.

    noise_duration :
        Total duration for which the noise time series should be
        generated.

    noise_dt :
        Time spacing of the generated frequency-noise realization.

    pulse_dt :
        Time step used to numerically integrate the atom's evolution
        during each finite-duration pulse.
    """

    def __init__(
        self,
        pulse_areas,
        pulse_duration,
        free_evolution_time,
        noise_model,
        noise_duration=None,
        noise_dt=1e-7,
        pulse_dt=1e-7,
    ):

        self.pulse_areas = np.asarray(pulse_areas, dtype=float)
        self.pulse_duration = float(pulse_duration)
        self.noise_model = noise_model

        self.noise_dt = float(noise_dt)
        self.pulse_dt = float(pulse_dt)

        # Free evolution

        if np.isscalar(free_evolution_time):
            self.free_times = np.full(
                len(self.pulse_areas) - 1,
                float(free_evolution_time),
            )
        else:
            self.free_times = np.asarray(
                free_evolution_time,
                dtype=float,
            )

            if len(self.free_times) != len(self.pulse_areas) - 1:
                raise ValueError(
                    "Need exactly N-1 free-evolution times for N pulses."
                )
            
        # Total experiment duration

        self.total_duration = (
            len(self.pulse_areas) * self.pulse_duration
            + np.sum(self.free_times)
        )

        if noise_duration is None:
            noise_duration = self.total_duration

        self.noise_duration = float(noise_duration)

        self.noise_t = None
        self.noise_hz = None

        self.times = []
        self.probabilities = []

    # Noise

    def generate_noise(self):

        t, noise = self.noise_model.generate_timeseries(
            duration=self.noise_duration,
            dt=self.noise_dt,
            n_realizations=1,
        )

        self.noise_t = np.asarray(t)
        self.noise_hz = np.asarray(noise[0])

    # Noise interpolation

    def generate_noise(self):
        """Generate ONE continuous laser-frequency-noise realization."""

        t, noise = self.noise_model.generate_timeseries(
        duration=self.noise_duration,
        dt=self.noise_dt,
        n_realizations=1,
        )

        self.noise_t = np.asarray(t)
        self.noise_hz = np.asarray(noise).squeeze()

        if self.noise_hz.ndim != 1:
            raise ValueError(
                f"Expected 1D frequency-noise array, "
                f"got shape {self.noise_hz.shape}"
            )

        if len(self.noise_t) != len(self.noise_hz):
            raise ValueError(
                "Noise time array and noise values have different lengths: "
                f"{len(self.noise_t)} vs {len(self.noise_hz)}"
            )

    def get_frequency_noise(self, t):

        if self.noise_t is None:
            raise RuntimeError(
            "Noise has not been generated. Call run()."
        )

        return np.interp(
        t,
        self.noise_t,
        self.noise_hz,
        left=self.noise_hz[0],
        right=self.noise_hz[-1],
    )
    # Hamiltonian

    @staticmethod
    def hamiltonian(Omega, detuning, phase=0.0):
        """
        Two-level Hamiltonian.

        H = 1/2 [ Delta        Omega*exp(-i*phase) ]
            [ Omega*exp(i*phase)   -Delta          ]

        where:
            Omega   = Rabi frequency [rad/s]
            Delta   = detuning [rad/s]
            phase   = drive/readout phase [rad]
        """

        return 0.5 * np.array(
            [
                [detuning, Omega * np.exp(-1j * phase)],
                [Omega * np.exp(1j * phase), -detuning],
            ],
            dtype=complex,
        )

    # Finite duration noisy pulse

    def apply_noisy_pulse(
        self,
        state,
        pulse_area,
        t_start,
        phase=0.0,
    ):
        """
        The pulse has a fixed resonant Rabi frequency chosen so that

            Omega * pulse_duration = pulse_area.

        During the pulse, the detuning changes according to the
        laser-frequency-noise realization.
        """

        tau = self.pulse_duration

        # Required resonant Rabi frequency
        Omega = pulse_area / tau

        # Number of integration steps
        n_steps = max(1, int(np.ceil(tau / self.pulse_dt)))

        dt = tau / n_steps

        state = np.asarray(state, dtype=complex)

        for i in range(n_steps):

            t = t_start + (i + 0.5) * dt

            delta_nu = self.get_frequency_noise(t)

            # Convert Hz -> rad/s
            Delta = 2.0 * np.pi * delta_nu

            H = self.hamiltonian(
                Omega=Omega,
                detuning=Delta,
                phase=phase,
            )

            U = expm(-1j * H * dt)

            state = U @ state

        return state

    # Free evolution

    def free_evolution(self, state, duration, t_start=None):
        """
        Free evolution with no additional Hamiltonian.

        This is intentionally trivial in the base class. We are
        isolating laser frequency noise, so gravity, Doppler, recoil,
        vibration, etc. are not included here.
        """

        return np.asarray(state, dtype=complex)

    # Full sequence

    def run(self, pulse_phase_offsets=None):
        """
        Run the entire pulse sequence.

        Parameters
        ----------
        pulse_phase_offsets :
            Drive-axis phase (rad) for each pulse, same length as
            `pulse_areas`. Defaults to all zeros. Set the last
            entry to pi/2 to read out at mid-fringe, where P_e responds
            linearly to accumulated phase noise instead of quadratically.

        Returns
        -------
        state :
            Final two-level state.

        probability :
            Final excited-state population.
        """

        if pulse_phase_offsets is None:
            pulse_phase_offsets = [0.0] * len(self.pulse_areas)
        elif len(pulse_phase_offsets) != len(self.pulse_areas):
            raise ValueError(
                "pulse_phase_offsets must match length of pulse_areas."
            )

        # Generate ONE noise realization
        self.generate_noise()

        # Initial state: |g> = [1, 0]
        state = np.array(
            [1.0 + 0j, 0.0 + 0j]
        )

        current_time = 0.0

        self.times = []
        self.probabilities = []

        # Execute pulse sequence
        for i, pulse_area in enumerate(self.pulse_areas):

            state = self.apply_noisy_pulse(
                state=state,
                pulse_area=pulse_area,
                t_start=current_time,
                phase=pulse_phase_offsets[i],
            )

            current_time += self.pulse_duration

            # Record state
            self.times.append(current_time)
            self.probabilities.append(
                np.abs(state[1]) ** 2
            )

            # Free evolution
            if i < len(self.pulse_areas) - 1:

                T = self.free_times[i]

                state = self.free_evolution(
                    state,
                    T,
                    t_start=current_time,
                )

                current_time += T

        probability = float(np.abs(state[1]) ** 2)

        return state, probability