import time
import numpy as np

from qitcat.modules.noise import FrequencyNoise
from pulse_sequence_frequency import FrequencyNoisePulseSequence
from N_pulse_sequence import (
    build_n_pulse_sequence,
    FastFrequencyNoisePulseSequence,
    run_single_shot,
)


def test_fast_matches_reference():
    """Same noise realization -> fast and reference propagators must agree."""
    pulse_areas, free_times, phase_offsets = build_n_pulse_sequence(
        n_pi_pulses=5,
        pulse_duration=20e-6,
        T_total=15e-3,
        readout_phase=np.pi / 2,
    )

    for seed in range(5):
        noise_ref = FrequencyNoise(ILW=864, f_k=1.16e6, noise_fraction=0.05, seed=seed)
        seq_ref = FrequencyNoisePulseSequence(
            pulse_areas=pulse_areas,
            pulse_duration=20e-6,
            free_evolution_time=free_times,
            noise_model=noise_ref,
            noise_dt=1e-7,
            pulse_dt=1e-7,
        )
        # Force identical noise realization by pre-seeding the model's RNG
        # the same way (both use noise_model's rng, so re-instantiate fresh
        # identical models per sequence to guarantee identical realizations).
        state_ref, prob_ref = seq_ref.run(pulse_phase_offsets=phase_offsets)

        noise_fast = FrequencyNoise(ILW=864, f_k=1.16e6, noise_fraction=0.05, seed=seed)
        seq_fast = FastFrequencyNoisePulseSequence(
            pulse_areas=pulse_areas,
            pulse_duration=20e-6,
            free_evolution_time=free_times,
            noise_model=noise_fast,
            noise_dt=1e-7,
            pulse_dt=1e-7,
        )
        state_fast, prob_fast = seq_fast.run(pulse_phase_offsets=phase_offsets)

        assert np.allclose(state_ref, state_fast, atol=1e-9), (
            f"seed={seed}: states differ: {state_ref} vs {state_fast}"
        )
        assert np.isclose(prob_ref, prob_fast, atol=1e-9), (
            f"seed={seed}: probs differ: {prob_ref} vs {prob_fast}"
        )
    print("PASS: fast propagator matches expm reference (5 seeds, N=5 pi-pulses)")


def test_speedup():
    noise_kwargs = dict(ILW=864, f_k=1.16e6, noise_fraction=0.05)

    t0 = time.time()
    for _ in range(5):
        run_single_shot(8, 10e-6, 20e-3, noise_kwargs, fast=False)
    t_ref = (time.time() - t0) / 5

    t0 = time.time()
    for _ in range(5):
        run_single_shot(8, 10e-6, 20e-3, noise_kwargs, fast=True)
    t_fast = (time.time() - t0) / 5

    print(f"reference: {t_ref:.4f} s/shot, fast: {t_fast:.4f} s/shot, "
          f"speedup: {t_ref/t_fast:.1f}x")


if __name__ == "__main__":
    test_fast_matches_reference()
    test_speedup()