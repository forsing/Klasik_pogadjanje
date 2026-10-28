#!/usr/bin/env python3



"""
QUANTUM POGADJANJE V2 — Loto 7/39
Python 3.11
Dependencies: numpy, scipy

Adaptation of the three-bit entanglement-assisted random access code.

MAIN COMPUTATION:
    1. Alice receives known historical occupancy bits.
    2. Her coding direction is controlled by learned rotations.
    3. Alice and Bob share a two-qubit singlet.
    4. Alice measures and sends one classical bit.
    5. Bob chooses the queried bit and its measurement basis.
    6. Joint amplitudes give the decoded Born probabilities.

The three-bit primitive is used separately for thirteen groups.
Each query uses a fresh preparation.
Different questions are not measured on one consumed resource.

LEARNED PARAMETERS:
    Alice's history-dependent rotation angles.
    Bob's orthogonal measurement frame for each group.

The fitted loss is the likelihood of complete seven-number sets.
No independent binary classifier is fitted.

No PCA.
No classical output-score regressor.
No mixture of classical distributions.
No random initialization, sampling or row shuffling.
No rounding or duplicate repair.
No cosmetic combination filters.

ADAPTATION TO LOTO:
    Each decoded query provides a Born probability q_i.
    Independent fresh protocol executions define a joint distribution.
    Condition that distribution on exactly seven positive outputs.

    P(S | history, count=7)
        = product(q_i for selected numbers)
          * product(1-q_i for unselected numbers)
          / P(count=7)

The original communication protocol reads encoded historical bits.
Training its coding and measurement settings for next-draw likelihood
is the temporal adaptation implemented here.

CHRONOLOGY:
    Train first 60%; validate next 10%.
    Train first 70%; validate next 10%.
    Select parameters using validation only.
    Train first 80%; evaluate final 20% once.
    Refit on ALL available historical transitions.

NEXT:
    Exact analytical maximum using Born probability odds.
    No search over 15 million combinations is necessary.
    Exhaustive small-case checks verify the maximum.

This is a local classical simulation of quantum amplitudes.
No Qiskit circuit or quantum hardware is executed.
"""



import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize



FILES = (
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
        4710,
    ),
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
        2977,
    ),
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
        1733,
    ),
)



NUMBER_COUNT = 39
DRAW_SIZE = 7
GROUP_COUNT = 13

CANDIDATES = (
    (1, 0.001),
    (1, 0.010),
    (3, 0.001),
    (3, 0.010),
)

INITIAL_ORDERS = (17, 11)

OPTIMIZER_STEPS = 600
OPTIMIZER_RESTARTS = 3
GRADIENT_LIMIT = 5e-4

IDENTITY = np.eye(2, dtype=np.complex128)

PAULI = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=np.complex128,
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rotate_vectors(vectors, angles, with_jacobian=False):
    """
    Apply Rx, then Ry, then Rz.
    Return analytical derivatives with respect to x/y/z angles.
    """
    vectors, angles = np.broadcast_arrays(
        np.asarray(vectors, dtype=np.float64),
        np.asarray(angles, dtype=np.float64),
    )

    require(
        vectors.shape[-1] == 3,
        "Rotacija zahteva trodimenzionalne vektore.",
    )
    require(
        np.all(np.isfinite(vectors))
        and np.all(np.isfinite(angles)),
        "Neispravan ulaz rotacije.",
    )

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    sx = np.sin(angles[..., 0])
    cx = np.cos(angles[..., 0])
    sy = np.sin(angles[..., 1])
    cy = np.cos(angles[..., 1])
    sz = np.sin(angles[..., 2])
    cz = np.cos(angles[..., 2])

    u0 = x
    u1 = cx * y - sx * z
    u2 = sx * y + cx * z

    v0 = cy * u0 + sy * u2
    v1 = u1
    v2 = -sy * u0 + cy * u2

    rotated = np.stack(
        (
            cz * v0 - sz * v1,
            sz * v0 + cz * v1,
            v2,
        ),
        axis=-1,
    )

    if not with_jacobian:
        return rotated, None

    dx_u1 = -sx * y - cx * z
    dx_u2 = cx * y - sx * z

    dx_v0 = sy * dx_u2
    dx_v1 = dx_u1
    dx_v2 = cy * dx_u2

    derivative_x = np.stack(
        (
            cz * dx_v0 - sz * dx_v1,
            sz * dx_v0 + cz * dx_v1,
            dx_v2,
        ),
        axis=-1,
    )

    dy_v0 = -sy * u0 + cy * u2
    dy_v2 = -cy * u0 - sy * u2

    derivative_y = np.stack(
        (
            cz * dy_v0,
            sz * dy_v0,
            dy_v2,
        ),
        axis=-1,
    )

    derivative_z = np.stack(
        (
            -sz * v0 - cz * v1,
            cz * v0 - sz * v1,
            np.zeros_like(v0),
        ),
        axis=-1,
    )

    jacobian = np.stack(
        (
            derivative_x,
            derivative_y,
            derivative_z,
        ),
        axis=-2,
    )

    return rotated, jacobian


def spinors_from_bloch(vectors):
    """
    Construct normalized complex amplitudes for unit Bloch vectors.

    Two coordinate charts avoid division by a vanishing amplitude.
    """
    vectors = np.asarray(vectors, dtype=np.float64)

    require(
        vectors.shape[-1] == 3
        and np.all(np.isfinite(vectors)),
        "Neispravan Blochov vektor.",
    )

    norm_error = np.max(
        np.abs(
            np.sum(vectors * vectors, axis=-1) - 1.0
        )
    )

    require(
        norm_error < 1e-8,
        "Blochov vektor nije jedinične dužine.",
    )

    original_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, 3)

    states = np.empty(
        (len(flat), 2),
        dtype=np.complex128,
    )

    north = flat[:, 2] >= 0
    south = ~north

    if np.any(north):
        selected = flat[north]
        alpha = np.sqrt((1.0 + selected[:, 2]) / 2.0)

        beta = (
            selected[:, 0] + 1j * selected[:, 1]
        ) / (2.0 * alpha)

        states[north, 0] = alpha
        states[north, 1] = beta

    if np.any(south):
        selected = flat[south]
        beta = np.sqrt((1.0 - selected[:, 2]) / 2.0)

        alpha = (
            selected[:, 0] - 1j * selected[:, 1]
        ) / (2.0 * beta)

        states[south, 0] = alpha
        states[south, 1] = beta

    norms = np.sqrt(
        np.sum(np.abs(states) ** 2, axis=1)
    )

    require(
        np.all(np.isfinite(norms))
        and np.all(norms > 0),
        "Kvantno stanje nema ispravnu normu.",
    )

    states /= norms[:, None]

    return states.reshape(original_shape + (2,))


def protocol_probabilities(
    alice_weights,
    bob_angles,
    x,
    latest_bits,
    with_jacobian=False,
):
    """
    Execute the EARAC probability calculation from singlet amplitudes.

    Alice:
        outcome A=0 for the positive eigenstate;
        outcome A=1 for the negative eigenstate;
        message M = first_input_bit XOR A.

    Bob:
        outcome B=0 for the negative eigenstate;
        outcome B=1 for the positive eigenstate;
        decoded bit = M XOR B.

    Joint amplitude:
        <a,b|singlet>
        = (conj(a0)*conj(b1) - conj(a1)*conj(b0)) / sqrt(2)
    """
    sample_count, number_count = latest_bits.shape

    require(
        number_count % 3 == 0,
        "Broj istorijskih oznaka mora biti deljiv sa tri.",
    )

    group_count = number_count // 3

    require(
        alice_weights.shape == (x.shape[1], group_count, 3)
        and bob_angles.shape == (group_count, 3),
        "Neispravan oblik parametara EARAC protokola.",
    )
    require(
        np.all((latest_bits == 0) | (latest_bits == 1)),
        "Istorijske oznake moraju biti nule ili jedinice.",
    )

    triples = latest_bits.reshape(
        sample_count,
        group_count,
        3,
    )

    baseline = (
        1.0 - 2.0 * triples
    ) / math.sqrt(3.0)

    alice_angles = np.einsum(
        "nf,fga->nga",
        x,
        alice_weights,
        optimize=True,
    )

    corrected_bloch, alice_jacobian = rotate_vectors(
        baseline,
        alice_angles,
        with_jacobian=with_jacobian,
    )

    query_axes = np.broadcast_to(
        np.eye(3),
        (group_count, 3, 3),
    )

    bob_angle_grid = np.broadcast_to(
        bob_angles[:, None, :],
        (group_count, 3, 3),
    )

    bob_axes, bob_jacobian = rotate_vectors(
        query_axes,
        bob_angle_grid,
        with_jacobian=with_jacobian,
    )

    # The first input bit determines Alice's basis convention.
    first_bit = triples[:, :, 0]
    first_sign = 1.0 - 2.0 * first_bit

    alice_axis = (
        first_sign[:, :, None] * corrected_bloch
    )

    alice_states = (
        spinors_from_bloch(alice_axis),
        spinors_from_bloch(-alice_axis),
    )

    # Bob's bit label: minus -> 0, plus -> 1.
    bob_states = (
        spinors_from_bloch(-bob_axes),
        spinors_from_bloch(bob_axes),
    )

    probability_one = np.zeros(
        (sample_count, group_count, 3),
        dtype=np.float64,
    )
    probability_zero = np.zeros_like(probability_one)

    for alice_outcome in (0, 1):
        alice_state = alice_states[alice_outcome]
        message = first_bit ^ alice_outcome

        a0 = np.conj(alice_state[:, :, 0])[:, :, None]
        a1 = np.conj(alice_state[:, :, 1])[:, :, None]

        for bob_outcome in (0, 1):
            bob_state = bob_states[bob_outcome]

            b0 = np.conj(bob_state[:, :, 0])[None, :, :]
            b1 = np.conj(bob_state[:, :, 1])[None, :, :]

            joint_amplitude = (
                a0 * b1 - a1 * b0
            ) / math.sqrt(2.0)

            joint_probability = (
                np.abs(joint_amplitude) ** 2
            )

            decoded = (
                message ^ bob_outcome
            )[:, :, None]

            probability_one += (
                joint_probability * decoded
            )
            probability_zero += (
                joint_probability * (1 - decoded)
            )

    total = probability_one + probability_zero

    require(
        np.all(np.isfinite(total))
        and np.max(np.abs(total - 1.0)) < 1e-10,
        "Zajedničke EARAC verovatnoće nisu normalizovane.",
    )

    # Remove floating-point normalization residue only.
    probability_one /= total
    probability_zero /= total

    decoded_bias = probability_zero - probability_one

    require(
        np.max(
            np.abs(
                np.sum(decoded_bias ** 2, axis=2) - 1.0
            )
        ) < 1e-8,
        "EARAC očitavanja ne zadovoljavaju kvantnu sfernu granicu.",
    )

    return (
        probability_one.reshape(sample_count, number_count),
        probability_zero.reshape(sample_count, number_count),
        corrected_bloch,
        alice_jacobian,
        bob_axes,
        bob_jacobian,
    )


def count_partition(
    probability_one,
    probability_zero,
    draw_size,
    with_gradient=False,
):
    """
    Compute P(exactly draw_size positive decoded outputs).

    Coefficient of z^draw_size in:
        product_i(probability_zero_i + probability_one_i*z)

    The derivative assumes p_zero = 1 - p_one.
    """
    q = np.asarray(probability_one, dtype=np.float64)
    r = np.asarray(probability_zero, dtype=np.float64)

    require(
        q.ndim == 2
        and q.shape == r.shape
        and q.shape[0] > 0
        and 1 <= draw_size <= q.shape[1],
        "Neispravan oblik uslovne raspodele.",
    )
    require(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(r))
        and np.all(q >= 0)
        and np.all(r >= 0)
        and np.max(np.abs(q + r - 1.0)) < 1e-10,
        "Neispravne Bornove verovatnoće.",
    )

    sample_count, number_count = q.shape

    forward = np.zeros(
        (sample_count, draw_size + 1),
        dtype=np.float64,
    )
    forward[:, 0] = 1.0

    cache = [forward] if with_gradient else None

    for number in range(number_count):
        following = (
            r[:, number, None] * forward
        )
        following[:, 1:] += (
            q[:, number, None] * forward[:, :-1]
        )

        forward = following

        if with_gradient:
            cache.append(forward)

    z = forward[:, draw_size]

    require(
        np.all(np.isfinite(z))
        and np.all(z > np.finfo(np.float64).tiny),
        "Događaj sa traženim brojem jedinica ima nultu "
        "ili numerički premalu verovatnoću.",
    )

    log_z = np.log(z)

    if not with_gradient:
        return log_z, None, None

    reverse = np.zeros_like(forward)
    reverse[:, draw_size] = 1.0 / z

    gradient_log_z = np.zeros_like(q)
    conditional_marginals = np.zeros_like(q)

    for number in range(number_count - 1, -1, -1):
        previous = cache[number]

        selected_term = np.sum(
            reverse[:, 1:] * previous[:, :-1],
            axis=1,
        )
        unselected_term = np.sum(
            reverse * previous,
            axis=1,
        )

        gradient_log_z[:, number] = (
            selected_term - unselected_term
        )

        conditional_marginals[:, number] = (
            q[:, number] * selected_term
        )

        previous_reverse = (
            r[:, number, None] * reverse
        )
        previous_reverse[:, :-1] += (
            q[:, number, None] * reverse[:, 1:]
        )

        reverse = previous_reverse

    require(
        np.all(np.isfinite(gradient_log_z))
        and np.all(np.isfinite(conditional_marginals)),
        "Izvod uslovne normalizacije nije konačan.",
    )
    require(
        np.min(conditional_marginals) >= -1e-8
        and np.max(conditional_marginals) <= 1.0 + 1e-8,
        "Uslovna uključna verovatnoća je van opsega [0,1].",
    )
    require(
        np.max(
            np.abs(
                conditional_marginals.sum(axis=1) - draw_size
            )
        ) < 1e-7,
        "Zbir uslovnih uključnih verovatnoća nije ispravan.",
    )

    return (
        log_z,
        gradient_log_z,
        conditional_marginals,
    )


def set_nll(q, r, targets, log_z):
    selected_probability = r.copy()

    selected_probability[
        np.arange(len(targets))[:, None],
        targets,
    ] = q[
        np.arange(len(targets))[:, None],
        targets,
    ]

    require(
        np.all(selected_probability > 0),
        "Posmatrano kolo ima nultu modelnu verovatnoću.",
    )

    nll = (
        -np.sum(np.log(selected_probability), axis=1)
        + log_z
    )

    require(
        np.all(np.isfinite(nll))
        and np.min(nll) >= -1e-7,
        "Neispravan gubitak kompletne kombinacije.",
    )

    return nll, selected_probability


def pack_parameters(alice_weights, bob_angles):
    return np.concatenate(
        (
            alice_weights.ravel(),
            bob_angles.ravel(),
        )
    )


def unpack_parameters(parameters, feature_count, group_count):
    size = feature_count * group_count * 3

    alice_weights = parameters[:size].reshape(
        feature_count,
        group_count,
        3,
    )
    bob_angles = parameters[size:].reshape(
        group_count,
        3,
    )

    return alice_weights, bob_angles


def objective(
    parameters,
    x,
    latest_bits,
    targets,
    ridge,
    draw_size,
):
    group_count = latest_bits.shape[1] // 3

    alice_weights, bob_angles = unpack_parameters(
        parameters,
        x.shape[1],
        group_count,
    )

    (
        q,
        r,
        corrected_bloch,
        alice_jacobian,
        bob_axes,
        bob_jacobian,
    ) = protocol_probabilities(
        alice_weights,
        bob_angles,
        x,
        latest_bits,
        with_jacobian=True,
    )

    log_z, gradient_log_z, _ = count_partition(
        q,
        r,
        draw_size,
        with_gradient=True,
    )

    nll, selected_probability = set_nll(
        q,
        r,
        targets,
        log_z,
    )

    observed = np.zeros_like(q)
    observed[
        np.arange(len(targets))[:, None],
        targets,
    ] = 1.0

    gradient_q = (
        gradient_log_z
        + (1.0 - 2.0 * observed) / selected_probability
    ) / len(targets)

    gradient_q = gradient_q.reshape(
        len(targets),
        group_count,
        3,
    )

    # Exact derivative of the decoded singlet Born probability:
    # q = (1 - corrected_bloch dot bob_axis) / 2.
    gradient_bloch = -0.5 * np.einsum(
        "ngj,gjc->ngc",
        gradient_q,
        bob_axes,
        optimize=True,
    )

    gradient_bob_axes = -0.5 * np.einsum(
        "ngj,ngc->gjc",
        gradient_q,
        corrected_bloch,
        optimize=True,
    )

    gradient_alice_angles = np.einsum(
        "ngc,ngac->nga",
        gradient_bloch,
        alice_jacobian,
        optimize=True,
    )

    gradient_alice_weights = np.einsum(
        "nf,nga->fga",
        x,
        gradient_alice_angles,
        optimize=True,
    )

    gradient_bob_angles = np.einsum(
        "gjc,gjac->ga",
        gradient_bob_axes,
        bob_jacobian,
        optimize=True,
    )

    penalty_mask = np.ones(x.shape[1], dtype=np.float64)
    penalty_mask[0] = 0.1

    penalty = ridge * (
        np.sum(
            alice_weights ** 2
            * penalty_mask[:, None, None]
        )
        + np.sum(bob_angles ** 2)
    )

    gradient_alice_weights += (
        2.0
        * ridge
        * alice_weights
        * penalty_mask[:, None, None]
    )

    gradient_bob_angles += (
        2.0 * ridge * bob_angles
    )

    loss = float(np.mean(nll) + penalty)

    gradient = pack_parameters(
        gradient_alice_weights,
        gradient_bob_angles,
    )

    require(
        np.isfinite(loss)
        and np.all(np.isfinite(gradient)),
        "Funkcija gubitka ili njen izvod nisu konačni.",
    )

    return loss, gradient


def initial_parameters(feature_count, group_count, order):
    weights = np.zeros(
        (feature_count, group_count, 3),
        dtype=np.float64,
    )

    phase = (
        2.0
        * np.pi
        * (order * np.arange(group_count) + 1)
        / (group_count + 1)
    )

    for axis in range(3):
        weights[0, :, axis] = (
            0.025 * np.sin(phase + 0.7 * axis)
        )

        if feature_count > 1:
            weights[1, :, axis] = (
                0.015 * np.cos(phase + 0.9 * axis)
            )

    bob_angles = np.column_stack(
        [
            0.02 * np.cos(phase + 0.5 * axis)
            for axis in range(3)
        ]
    )

    return pack_parameters(weights, bob_angles)


def load_draws(path, expected_rows):
    fingerprint = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rows = []
    blank_rows = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(not item.strip() for item in row):
                blank_rows += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                values = [int(item.strip()) for item in row]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == DRAW_SIZE,
                f"{path.name}, red {line_number}: ponovljen broj.",
            )
            require(
                all(1 <= value <= NUMBER_COUNT for value in values),
                f"{path.name}, red {line_number}: broj van opsega 1–39.",
            )

            rows.append(
                sorted(value - 1 for value in values)
            )

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, pročitano {len(rows)}.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    require(
        draws.shape == (expected_rows, DRAW_SIZE),
        "Neispravan oblik istorije.",
    )

    occupancy = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.int64,
    )
    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank_rows}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def make_context(occupancy, target_indices, lookback):
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    require(
        target_indices.ndim == 1
        and np.all(target_indices >= 1)
        and np.all(target_indices <= len(occupancy)),
        "Neispravni ciljni indeksi.",
    )

    number_count = occupancy.shape[1]
    block_size = number_count + 1

    raw = np.zeros(
        (len(target_indices), lookback * block_size),
        dtype=np.float64,
    )

    for lag in range(1, lookback + 1):
        previous = target_indices - lag
        valid = np.flatnonzero(previous >= 0)
        offset = (lag - 1) * block_size

        if valid.size:
            raw[
                valid,
                offset:offset + number_count,
            ] = occupancy[previous[valid]]

            raw[
                valid,
                offset + number_count,
            ] = 1.0

    latest_bits = occupancy[
        target_indices - 1
    ].copy()

    return raw, latest_bits


def fit_scaling(raw):
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)

    scale = scale.copy()
    scale[scale < 1e-12] = 1.0

    return mean, scale


def transform_context(raw, mean, scale):
    standardized = (raw - mean) / scale

    return np.column_stack(
        (
            np.ones(len(raw)),
            standardized,
        )
    )


@dataclass
class Model:
    alice_weights: np.ndarray
    bob_angles: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    lookback: int
    ridge: float
    train_loss: float
    gradient_norm: float

    def transform(self, raw):
        return transform_context(
            raw,
            self.mean,
            self.scale,
        )

    def probabilities(self, raw, latest_bits):
        result = protocol_probabilities(
            self.alice_weights,
            self.bob_angles,
            self.transform(raw),
            latest_bits,
        )
        return result[0], result[1]


def fit_model(
    raw,
    latest_bits,
    targets,
    lookback,
    ridge,
    label,
):
    mean, scale = fit_scaling(raw)
    x = transform_context(raw, mean, scale)

    group_count = latest_bits.shape[1] // 3
    fitted = []

    for order in INITIAL_ORDERS:
        say(
            f"{label}: lag={lookback}, lambda={ridge}, "
            f"početak={order}, trening={len(targets)}"
        )

        parameters = initial_parameters(
            x.shape[1],
            group_count,
            order,
        )

        iteration = 0

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 100 == 0:
                say(
                    f"  Optimizacija geometrije: {iteration} iteracija"
                )

        def function(current):
            return objective(
                current,
                x,
                latest_bits,
                targets,
                ridge,
                DRAW_SIZE,
            )

        result = None
        gradient_norm = math.inf
        converged = False

        try:
            for _ in range(OPTIMIZER_RESTARTS):
                result = minimize(
                    function,
                    parameters,
                    method="L-BFGS-B",
                    jac=True,
                    callback=callback,
                    options={
                        "maxiter": OPTIMIZER_STEPS,
                        "maxls": 60,
                        "maxcor": 20,
                        "ftol": 1e-13,
                        "gtol": 1e-6,
                    },
                )

                parameters = result.x.copy()
                loss, gradient = function(parameters)

                gradient_norm = float(
                    np.max(np.abs(gradient))
                )

                if (
                    result.success
                    and gradient_norm <= GRADIENT_LIMIT
                ):
                    converged = True
                    break

            if not converged:
                message = (
                    result.message
                    if result is not None
                    else "bez rezultata"
                )

                say(
                    f"  Početak nije prihvaćen: {message}; "
                    f"max|grad|={gradient_norm:.3e}"
                )
                continue

            alice_weights, bob_angles = unpack_parameters(
                parameters,
                x.shape[1],
                group_count,
            )

            fitted.append(
                Model(
                    alice_weights=alice_weights.copy(),
                    bob_angles=bob_angles.copy(),
                    mean=mean.copy(),
                    scale=scale.copy(),
                    lookback=lookback,
                    ridge=ridge,
                    train_loss=float(loss),
                    gradient_norm=gradient_norm,
                )
            )

            say(
                f"  Prihvaćeno: gubitak={loss:.8f}, "
                f"max|grad|={gradient_norm:.3e}"
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"  Početak nije prihvaćen: {error}"
            )

    require(
        bool(fitted),
        f"{label}: nijedan početak nije prošao numeričke uslove.",
    )

    return min(
        fitted,
        key=lambda model: model.train_loss,
    )


def evaluate_nll(model, raw, latest_bits, targets):
    total = 0.0

    for start in range(0, len(targets), 512):
        stop = min(start + 512, len(targets))

        q, r = model.probabilities(
            raw[start:stop],
            latest_bits[start:stop],
        )

        log_z = count_partition(
            q,
            r,
            DRAW_SIZE,
        )[0]

        nll = set_nll(
            q,
            r,
            targets[start:stop],
            log_z,
        )[0]

        total += float(np.sum(nll))

    return total / len(targets)


def verify_full_count_distribution(q, r, draw_size, log_z):
    number_count = len(q)

    distribution = np.zeros(
        number_count + 1,
        dtype=np.float64,
    )
    distribution[0] = 1.0

    for number in range(number_count):
        following = r[number] * distribution
        following[1:] += (
            q[number] * distribution[:-1]
        )
        distribution = following

    require(
        np.all(np.isfinite(distribution))
        and np.min(distribution) >= 0,
        "Neispravna puna raspodela broja jedinica.",
    )
    require(
        abs(float(distribution.sum()) - 1.0) < 1e-10,
        "Puna raspodela broja jedinica nije normalizovana.",
    )

    expected_z = math.exp(float(log_z))

    require(
        abs(
            float(distribution[draw_size]) - expected_z
        ) <= 1e-10 * expected_z,
        "Dve konstrukcije normalizacione konstante nisu saglasne.",
    )


def exact_next(q, r, draw_size):
    """
    Exact maximum of the conditional distribution.

    Select every forced q=1 outcome.
    Exclude every q=0 outcome.
    Fill the remaining positions by decreasing log(q/r).
    Exact ties use the lower number.
    """
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    log_z = float(
        count_partition(
            q[None, :],
            r[None, :],
            draw_size,
        )[0][0]
    )

    forced = np.flatnonzero(r == 0)
    interior = np.flatnonzero(
        (q > 0) & (r > 0)
    )

    required = draw_size - len(forced)

    require(
        required >= 0 and required <= len(interior),
        "Nema legalne kombinacije sa traženom veličinom.",
    )

    odds = (
        np.log(q[interior])
        - np.log(r[interior])
    )

    order = np.lexsort(
        (interior, -odds)
    )

    chosen_interior = interior[
        order[:required]
    ]

    selected = np.sort(
        np.concatenate(
            (forced, chosen_interior)
        )
    )

    require(
        len(selected) == draw_size
        and len(set(selected.tolist())) == draw_size,
        "Analitički maksimum nema ispravnu veličinu.",
    )

    if 0 < required < len(interior):
        chosen_minimum = float(
            np.min(odds[order[:required]])
        )
        rejected_maximum = float(
            np.max(odds[order[required:]])
        )

        require(
            chosen_minimum >= rejected_maximum,
            "Izbor odnosa verovatnoća nije maksimalan.",
        )

    selected_probability = r.copy()
    selected_probability[selected] = q[selected]

    require(
        np.all(selected_probability > 0),
        "Izabrana kombinacija ima nultu verovatnoću.",
    )

    log_probability = (
        float(np.sum(np.log(selected_probability)))
        - log_z
    )

    require(
        np.isfinite(log_probability)
        and log_probability <= 1e-7,
        "Neispravna verovatnoća maksimuma.",
    )

    verify_full_count_distribution(
        q,
        r,
        draw_size,
        log_z,
    )

    return (
        tuple(int(value) for value in selected),
        log_probability,
        log_z,
    )


def verify_protocol_by_projectors(
    latest_bits,
    corrected_bloch,
    bob_axes,
    probability_one,
):
    singlet = np.array(
        [0.0, 1.0, -1.0, 0.0],
        dtype=np.complex128,
    ) / math.sqrt(2.0)

    for sample in range(len(latest_bits)):
        first_bit = int(latest_bits[sample, 0])

        alice_axis = (
            (1.0 - 2.0 * first_bit)
            * corrected_bloch[sample, 0]
        )

        alice_operator = np.einsum(
            "a,aij->ij",
            alice_axis,
            PAULI,
        )

        for query in range(3):
            bob_operator = np.einsum(
                "a,aij->ij",
                bob_axes[0, query],
                PAULI,
            )

            total = 0.0
            decoded_one = 0.0

            for alice_outcome in (0, 1):
                alice_sign = (
                    1.0 if alice_outcome == 0 else -1.0
                )

                alice_projector = (
                    IDENTITY + alice_sign * alice_operator
                ) / 2.0

                message = first_bit ^ alice_outcome

                for bob_outcome in (0, 1):
                    bob_sign = (
                        -1.0 if bob_outcome == 0 else 1.0
                    )

                    bob_projector = (
                        IDENTITY + bob_sign * bob_operator
                    ) / 2.0

                    probability_complex = np.vdot(
                        singlet,
                        np.kron(
                            alice_projector,
                            bob_projector,
                        ) @ singlet,
                    )

                    require(
                        abs(probability_complex.imag) < 1e-12,
                        "Projektorska verovatnoća ima imaginarni ostatak.",
                    )

                    probability = float(
                        probability_complex.real
                    )

                    require(
                        probability >= -1e-12,
                        "Projektorska verovatnoća je negativna.",
                    )

                    total += probability

                    if message ^ bob_outcome:
                        decoded_one += probability

            require(
                abs(total - 1.0) < 1e-12
                and abs(
                    decoded_one - probability_one[sample, query]
                ) < 1e-12,
                "Singlet amplitude i nezavisni projektorski račun "
                "nisu saglasni.",
            )


def check_quantum_protocol():
    latest = np.array(
        [
            [
                (code >> 2) & 1,
                (code >> 1) & 1,
                code & 1,
            ]
            for code in range(8)
        ],
        dtype=np.int64,
    )

    x = np.column_stack(
        (
            np.ones(8),
            np.arange(8) / 7.0,
        )
    )

    weights = np.array(
        [
            [[0.12, -0.08, 0.09]],
            [[0.03, 0.04, -0.02]],
        ],
        dtype=np.float64,
    )

    bob = np.array(
        [[0.10, -0.05, 0.07]],
        dtype=np.float64,
    )

    result = protocol_probabilities(
        weights,
        bob,
        x,
        latest,
        with_jacobian=True,
    )

    q, r, corrected, _, bob_axes, _ = result

    verify_protocol_by_projectors(
        latest,
        corrected,
        bob_axes,
        q,
    )

    expected_q = (
        1.0
        - np.einsum(
            "ngc,gjc->ngj",
            corrected,
            bob_axes,
        ).reshape(8, 3)
    ) / 2.0

    require(
        np.max(np.abs(q - expected_q)) < 1e-12,
        "Bornove amplitude i korelacioni izraz nisu saglasni.",
    )

    baseline = protocol_probabilities(
        np.zeros_like(weights),
        np.zeros_like(bob),
        x,
        latest,
    )

    baseline_q, baseline_r = baseline[:2]

    success = np.where(
        latest == 1,
        baseline_q,
        baseline_r,
    )

    expected_success = (
        0.5 * (1.0 + 1.0 / math.sqrt(3.0))
    )

    require(
        np.max(
            np.abs(success - expected_success)
        ) < 1e-12,
        "Osnovni trobitni EARAC protokol nije prošao proveru.",
    )


def self_checks():
    say(
        "",
        "Pokrećem matematičke i numeričke samoprovere."
    )

    check_quantum_protocol()

    x = np.array(
        [
            [1.0, 0.2, -0.4],
            [1.0, -0.3, 0.1],
            [1.0, 0.7, 0.5],
            [1.0, -0.6, -0.2],
        ],
        dtype=np.float64,
    )

    latest = np.array(
        [
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [0, 3],
            [1, 5],
            [2, 4],
            [0, 5],
        ],
        dtype=np.int64,
    )

    draw_size = 2

    parameters = initial_parameters(
        x.shape[1],
        2,
        order=2,
    )

    _, analytic = objective(
        parameters,
        x,
        latest,
        targets,
        0.01,
        draw_size,
    )

    numerical = np.empty_like(parameters)
    epsilon = 1e-6

    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        f_plus = objective(
            plus,
            x,
            latest,
            targets,
            0.01,
            draw_size,
        )[0]

        f_minus = objective(
            minus,
            x,
            latest,
            targets,
            0.01,
            draw_size,
        )[0]

        numerical[index] = (
            (f_plus - f_minus)
            / (2.0 * epsilon)
        )

    gradient_error = float(
        np.max(
            np.abs(analytic - numerical)
            / (
                1.0
                + np.abs(analytic)
                + np.abs(numerical)
            )
        )
    )

    require(
        gradient_error < 2e-5,
        f"Provera izvoda nije prošla: {gradient_error:.3e}.",
    )

    weights, bob = unpack_parameters(
        parameters,
        x.shape[1],
        2,
    )

    q, r = protocol_probabilities(
        weights,
        bob,
        x[:1],
        latest[:1],
    )[:2]

    for test_q, test_r in (
        (q[0], r[0]),
        (
            np.array([1.0, 0.0, 0.2, 0.4, 0.7, 0.6]),
            np.array([0.0, 1.0, 0.8, 0.6, 0.3, 0.4]),
        ),
    ):
        all_sets = list(
            combinations(range(len(test_q)), draw_size)
        )

        unnormalized = []

        for selected in all_sets:
            probabilities = test_r.copy()
            indices = np.asarray(selected, dtype=np.int64)
            probabilities[indices] = test_q[indices]

            unnormalized.append(
                float(np.prod(probabilities))
            )

        unnormalized = np.asarray(unnormalized)
        direct_z = float(unnormalized.sum())

        log_z = float(
            count_partition(
                test_q[None, :],
                test_r[None, :],
                draw_size,
            )[0][0]
        )

        require(
            abs(math.exp(log_z) - direct_z) < 1e-12,
            "Rekurzivna normalizacija nije saglasna sa nabrajanjem.",
        )

        normalized = unnormalized / direct_z

        require(
            abs(float(normalized.sum()) - 1.0) < 1e-12,
            "Mala uslovna raspodela nije normalizovana.",
        )

        best, best_log_probability, _ = exact_next(
            test_q,
            test_r,
            draw_size,
        )

        best_index = all_sets.index(best)

        require(
            abs(
                math.exp(best_log_probability)
                - float(np.max(normalized))
            ) < 1e-12
            and abs(
                math.exp(best_log_probability)
                - float(normalized[best_index])
            ) < 1e-12,
            "Analitički maksimum nije prošao potpunu malu proveru.",
        )

    # Check the actual seven-number normalization.
    reference_q = (
        0.2 + 0.05 * np.sin(np.arange(9))
    )
    reference_r = 1.0 - reference_q

    reference_log_z = float(
        count_partition(
            reference_q[None, :],
            reference_r[None, :],
            7,
        )[0][0]
    )

    direct_z = 0.0

    for selected in combinations(range(9), 7):
        probabilities = reference_r.copy()
        indices = np.asarray(selected, dtype=np.int64)
        probabilities[indices] = reference_q[indices]
        direct_z += float(np.prod(probabilities))

    require(
        abs(
            math.exp(reference_log_z) - direct_z
        ) < 1e-12,
        "Normalizacija za sedam brojeva nije prošla proveru.",
    )

    raw, previous = make_context(
        latest,
        [1],
        lookback=3,
    )

    block_size = latest.shape[1] + 1

    require(
        np.all(raw[0, :latest.shape[1]] == latest[0])
        and raw[0, block_size - 1] == 1.0
        and np.all(raw[0, block_size:] == 0.0)
        and np.all(previous[0] == latest[0]),
        "Početni istorijski prelaz nije ispravan.",
    )

    say(
        "Samoprovere prošle: singlet amplitude, poruka i merenje,",
        "kvantna geometrija, izvod, normalizacija i tačan maksimum.",
        f"Relativna greška izvoda: {gradient_error:.3e}",
        "Provera osnovnog EARAC protokola odnosi se na poznate bitove.",
    )


def longest_consecutive_run(combination):
    longest = 1
    current = 1

    for left, right in zip(
        combination,
        combination[1:],
    ):
        if right == left + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def process_file(path, draws, occupancy, fingerprint):
    targets = draws[1:].copy()
    sample_count = len(targets)

    train_60 = int(0.60 * sample_count)
    train_70 = int(0.70 * sample_count)
    train_80 = int(0.80 * sample_count)

    require(
        50 < train_60 < train_70 < train_80 < sample_count,
        "Premalo podataka za hronološku podelu.",
    )

    contexts = {}

    for lookback, _ in CANDIDATES:
        if lookback not in contexts:
            contexts[lookback] = make_context(
                occupancy,
                np.arange(
                    1,
                    len(draws),
                    dtype=np.int64,
                ),
                lookback,
            )

    folds = (
        (train_60, train_70),
        (train_70, train_80),
    )

    scores = []

    for lookback, ridge in CANDIDATES:
        raw, latest = contexts[lookback]

        total_nll = 0.0
        total_count = 0

        for fold_number, (
            train_end,
            validation_end,
        ) in enumerate(folds, start=1):
            model = fit_model(
                raw[:train_end],
                latest[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                label=f"{path.name} / validacija {fold_number}",
            )

            count = validation_end - train_end

            nll = evaluate_nll(
                model,
                raw[train_end:validation_end],
                latest[train_end:validation_end],
                targets[train_end:validation_end],
            )

            total_nll += count * nll
            total_count += count

            say(
                f"  Validacioni NLL po kolu: {nll:.8f}"
            )

        score = total_nll / total_count
        scores.append((score, lookback, ridge))

        say(
            f"KANDIDAT: lag={lookback}, lambda={ridge}, "
            f"validacioni NLL={score:.8f}"
        )

    validation_nll, selected_lookback, selected_ridge = min(
        scores,
        key=lambda item: item[0],
    )

    raw, latest = contexts[selected_lookback]

    say(
        "",
        f"IZABRANO: lag={selected_lookback}, "
        f"lambda={selected_ridge}, "
        f"validacioni NLL={validation_nll:.8f}",
    )

    test_model = fit_model(
        raw[:train_80],
        latest[:train_80],
        targets[:train_80],
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        raw[train_80:],
        latest[train_80:],
        targets[train_80:],
    )

    say(
        f"ODVOJENI TEST: {sample_count - train_80} kola, "
        f"NLL po kolu={test_nll:.8f}",
        "Test se ne koristi za promenu parametara.",
    )

    final_model = fit_model(
        raw,
        latest,
        targets,
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / završno učenje",
    )

    say(
        f"ZAVRŠNO UČENJE: svih {sample_count} prelaza "
        f"iz {len(draws)} kola."
    )

    raw_next, latest_next = make_context(
        occupancy,
        [len(draws)],
        selected_lookback,
    )

    q_next, r_next = final_model.probabilities(
        raw_next,
        latest_next,
    )

    log_z, _, next_marginals = count_partition(
        q_next,
        r_next,
        DRAW_SIZE,
        with_gradient=True,
    )

    best, log_probability, maximum_log_z = exact_next(
        q_next[0],
        r_next[0],
        DRAW_SIZE,
    )

    require(
        abs(float(log_z[0]) - maximum_log_z) < 1e-10,
        "Normalizacija i izračunavanje maksimuma nisu saglasni.",
    )

    # Compare normalized inclusion marginals on actual prior contexts.
    q_previous, r_previous = final_model.probabilities(
        raw[-8:],
        latest[-8:],
    )

    previous_marginals = count_partition(
        q_previous,
        r_previous,
        DRAW_SIZE,
        with_gradient=True,
    )[2]

    sensitivity = float(
        np.max(
            np.abs(
                previous_marginals - next_marginals
            )
        )
    )

    require(
        len(best) == DRAW_SIZE
        and len(set(best)) == DRAW_SIZE
        and all(0 <= value < NUMBER_COUNT for value in best),
        "NEXT nije legalna kombinacija 7/39.",
    )

    run_length = longest_consecutive_run(best)

    repetitions = int(
        np.count_nonzero(
            np.all(
                draws == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    warnings = []

    if run_length >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run_length} uzastopnih brojeva. "
            "Potreban je pregled modela. Rezultat nije filtriran."
        )

    if repetitions:
        warnings.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji. "
            "Rezultat nije promenjen."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Na proverenim prethodnim kontekstima nije utvrđena "
            "promena uslovnih uključnih verovatnoća. "
            "Proveriti uticaj istorije."
        )

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "combination": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "model_probability": math.exp(log_probability),
        "normalizer": math.exp(maximum_log_z),
        "marginal_sum": float(next_marginals.sum()),
        "sensitivity": sensitivity,
        "run_length": run_length,
        "span": best[-1] - best[0],
        "warnings": warnings,
    }


def main():
    np.seterr(
        over="raise",
        invalid="raise",
        divide="raise",
    )

    require(
        NUMBER_COUNT == GROUP_COUNT * 3,
        "Broj oznaka nije saglasan sa trobitnim protokolom.",
    )

    say(
        "QUANTUM POGADJANJE V2 — Loto 7/39",
        "Spregnuti singlet, Aliceina poruka i Bobovo merenje.",
        "Učenje uglova kodiranja i mernih baza.",
        "Bornove verovatnoće iz kompleksnih amplituda.",
        "Uslovna raspodela kompletnih kombinacija od sedam brojeva.",
        "Lokalni klasični simulator kvantnog protokola.",
        "Bez slučajnog generisanja i bez klasifikatora.",
        "NEXT koristi tačan analitički maksimum.",
    )

    resolved_paths = [
        path.resolve()
        for path, _ in FILES
    ]

    require(
        len(set(resolved_paths)) == len(FILES),
        "Putanje CSV-ova nisu različite.",
    )

    datasets = []

    for path, expected_rows in FILES:
        draws, occupancy, fingerprint = load_draws(
            path,
            expected_rows,
        )

        datasets.append(
            (path, draws, occupancy, fingerprint)
        )

    require(
        len(
            {
                fingerprint
                for _, _, _, fingerprint in datasets
            }
        ) == 3,
        "Dva CSV-a imaju identičan sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, occupancy, fingerprint in datasets:
        results.append(
            process_file(
                path,
                draws,
                occupancy,
                fingerprint,
            )
        )

    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            if (
                results[left]["combination"]
                == results[right]["combination"]
            ):
                message = (
                    "Ista NEXT kombinacija za različite CSV-ove: "
                    f"{results[left]['name']} i "
                    f"{results[right]['name']}. "
                    "Proveriti uticaj istorije i moguću degeneraciju. "
                    "Kombinacija nije promenjena."
                )

                results[left]["warnings"].append(message)
                results[right]["warnings"].append(message)

    say(
        "",
        "============================================================",
        "ZAVRŠNI REZULTATI QUANTUM POGADJANJE V2",
        "============================================================",
        "NLL koristi prirodni logaritam.",
        "Niži NLL označava bolji rezultat na tom skupu.",
        "Modelna verovatnoća pripada naučenom modelu.",
        "Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.",
    )

    for result in results:
        numbers = " ".join(
            f"{value:02d}"
            for value in result["combination"]
        )

        say(
            "",
            f"CSV: {result['name']}",
            f"NEXT: {numbers}",
            f"Validacioni NLL: {result['validation_nll']:.8f}",
            f"Odvojeni test NLL: {result['test_nll']:.8f}",
            "Modelna verovatnoća NEXT: "
            f"{result['model_probability']:.12g}",
            "Normalizaciona konstanta P(broj jedinica=7): "
            f"{result['normalizer']:.12g}",
            "Zbir uslovnih uključnih verovatnoća: "
            f"{result['marginal_sum']:.12f}",
            "Promena uslovnih uključnih verovatnoća kroz kontekst: "
            f"{result['sensitivity']:.6e}",
            f"Najduži uzastopni niz: {result['run_length']}",
            f"Raspon kombinacije: {result['span']}",
            "Izbor NEXT: tačan maksimum uslovne raspodele.",
        )

        for warning in result["warnings"]:
            say(
                f"PROVERA MODELA: {warning}"
            )

    say(
        "",
        "Učenje koristi lokalnu numeričku optimizaciju.",
        "Globalni optimum parametara nije dokazan.",
        "Zavisnosti izlaza ograničene su konstrukcijom protokola",
        "i uslovom da kombinacija sadrži tačno sedam brojeva.",
        "Simulacija ne dokazuje računarsku kvantnu prednost.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        print(
            f"ZAUSTAVLJENO: {error}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Završni skup od tri NEXT rezultata nije potvrđen.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Tačno mesto prekida:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)




"""
QUANTUM POGADJANJE V2 — Loto 7/39
Spregnuti singlet, Aliceina poruka i Bobovo merenje.
Učenje uglova kodiranja i mernih baza.
Bornove verovatnoće iz kompleksnih amplituda.
Uslovna raspodela kompletnih kombinacija od sedam brojeva.
Lokalni klasični simulator kvantnog protokola.
Bez slučajnog generisanja i bez klasifikatora.
NEXT koristi tačan analitički maksimum.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem matematičke i numeričke samoprovere.
Samoprovere prošle: singlet amplitude, poruka i merenje,
kvantna geometrija, izvod, normalizacija i tačan maksimum.
Relativna greška izvoda: 3.857e-10
Provera osnovnog EARAC protokola odnosi se na poznate bitove.
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.001, početak=17, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.29454471, max|grad|=8.327e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.001, početak=11, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.29454471, max|grad|=9.052e-07
  Validacioni NLL po kolu: 16.96975490
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.001, početak=17, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.34428508, max|grad|=9.814e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.001, početak=11, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.34428508, max|grad|=8.787e-07
  Validacioni NLL po kolu: 16.85865175
KANDIDAT: lag=1, lambda=0.001, validacioni NLL=16.91420333
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.01, početak=17, trening=2825
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.55639240, max|grad|=8.257e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.01, početak=11, trening=2825
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.55639240, max|grad|=8.322e-07
  Validacioni NLL po kolu: 16.96064706
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.01, početak=17, trening=3296
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.59903710, max|grad|=9.171e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.01, početak=11, trening=3296
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.59903710, max|grad|=7.240e-07
  Validacioni NLL po kolu: 16.90148144
KANDIDAT: lag=1, lambda=0.01, validacioni NLL=16.93106425
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.001, početak=17, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Prihvaćeno: gubitak=14.87509544, max|grad|=9.544e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.001, početak=11, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Prihvaćeno: gubitak=14.83211705, max|grad|=6.198e-07
  Validacioni NLL po kolu: 20.56661734
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.001, početak=17, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Prihvaćeno: gubitak=15.25655172, max|grad|=8.710e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.001, početak=11, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Prihvaćeno: gubitak=15.28546704, max|grad|=7.174e-07
  Validacioni NLL po kolu: 19.96052320
KANDIDAT: lag=3, lambda=0.001, validacioni NLL=20.26357027
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.01, početak=17, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Prihvaćeno: gubitak=15.56911159, max|grad|=4.630e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.01, početak=11, trening=2825
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Prihvaćeno: gubitak=15.52737274, max|grad|=8.261e-07
  Validacioni NLL po kolu: 19.95931156
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.01, početak=17, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Prihvaćeno: gubitak=15.90555457, max|grad|=8.403e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.01, početak=11, trening=3296
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Prihvaćeno: gubitak=15.91214861, max|grad|=7.108e-07
  Validacioni NLL po kolu: 19.06608638
KANDIDAT: lag=3, lambda=0.01, validacioni NLL=19.51269897

IZABRANO: lag=1, lambda=0.001, validacioni NLL=16.91420333
loto7_4710_k86.csv / odvojeni test: lag=1, lambda=0.001, početak=17, trening=3767
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.37551772, max|grad|=9.367e-07
loto7_4710_k86.csv / odvojeni test: lag=1, lambda=0.001, početak=11, trening=3767
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.37551772, max|grad|=8.472e-07
ODVOJENI TEST: 942 kola, NLL po kolu=16.79439308
Test se ne koristi za promenu parametara.
loto7_4710_k86.csv / završno učenje: lag=1, lambda=0.001, početak=17, trening=4709
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.41815477, max|grad|=8.223e-07
loto7_4710_k86.csv / završno učenje: lag=1, lambda=0.001, početak=11, trening=4709
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.41815477, max|grad|=9.923e-07
ZAVRŠNO UČENJE: svih 4709 prelaza iz 4710 kola.
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.001, početak=17, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.03561741, max|grad|=9.092e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.001, početak=11, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=16.05608660, max|grad|=7.927e-07
  Validacioni NLL po kolu: 17.31740173
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.001, početak=17, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.17394711, max|grad|=7.961e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.001, početak=11, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=16.16415214, max|grad|=7.907e-07
  Validacioni NLL po kolu: 17.17316579
KANDIDAT: lag=1, lambda=0.001, validacioni NLL=17.24540497
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.01, početak=17, trening=1785
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.33447560, max|grad|=8.930e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.01, početak=11, trening=1785
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.33447560, max|grad|=9.628e-07
  Validacioni NLL po kolu: 17.20402146
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.01, početak=17, trening=2083
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.39858824, max|grad|=7.175e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.01, početak=11, trening=2083
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.39858824, max|grad|=6.908e-07
  Validacioni NLL po kolu: 17.18357221
KANDIDAT: lag=1, lambda=0.01, validacioni NLL=17.19381402
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.001, početak=17, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Prihvaćeno: gubitak=12.30498925, max|grad|=8.739e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.001, početak=11, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Optimizacija geometrije: 1200 iteracija
  Optimizacija geometrije: 1300 iteracija
  Optimizacija geometrije: 1400 iteracija
  Optimizacija geometrije: 1500 iteracija
  Prihvaćeno: gubitak=12.35447552, max|grad|=9.083e-07
  Validacioni NLL po kolu: 24.02415134
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.001, početak=17, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Optimizacija geometrije: 1200 iteracija
  Optimizacija geometrije: 1300 iteracija
  Prihvaćeno: gubitak=13.29566744, max|grad|=6.055e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.001, početak=11, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=13.26232672, max|grad|=8.386e-07
  Validacioni NLL po kolu: 24.35283066
KANDIDAT: lag=3, lambda=0.001, validacioni NLL=24.18821480
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.01, početak=17, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=13.57894217, max|grad|=9.785e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.01, početak=11, trening=1785
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=13.62963612, max|grad|=6.107e-07
  Validacioni NLL po kolu: 24.08222048
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.01, početak=17, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Optimizacija geometrije: 1200 iteracija
  Prihvaćeno: gubitak=14.64120088, max|grad|=7.894e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.01, početak=11, trening=2083
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Prihvaćeno: gubitak=14.79225737, max|grad|=7.736e-07
  Validacioni NLL po kolu: 22.82223749
KANDIDAT: lag=3, lambda=0.01, validacioni NLL=23.45328780

IZABRANO: lag=1, lambda=0.01, validacioni NLL=17.19381402
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=1, lambda=0.01, početak=17, trening=2380
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.45346499, max|grad|=9.729e-07
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=1, lambda=0.01, početak=11, trening=2380
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.45346499, max|grad|=9.354e-07
ODVOJENI TEST: 596 kola, NLL po kolu=17.18068961
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_2977.csv / završno učenje: lag=1, lambda=0.01, početak=17, trening=2976
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.54850873, max|grad|=7.966e-07
loto7_4710_k86_loto_2977.csv / završno učenje: lag=1, lambda=0.01, početak=11, trening=2976
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.54850873, max|grad|=7.734e-07
ZAVRŠNO UČENJE: svih 2976 prelaza iz 2977 kola.
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.001, početak=17, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=15.31349318, max|grad|=9.769e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.001, početak=11, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Prihvaćeno: gubitak=15.34553233, max|grad|=8.167e-07
  Validacioni NLL po kolu: 20.25976413
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.001, početak=17, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Prihvaćeno: gubitak=15.58523961, max|grad|=9.187e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.001, početak=11, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Prihvaćeno: gubitak=15.57817277, max|grad|=8.196e-07
  Validacioni NLL po kolu: 19.03606978
KANDIDAT: lag=1, lambda=0.001, validacioni NLL=19.64791696
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.01, početak=17, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=15.76053101, max|grad|=7.425e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.01, početak=11, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=15.76407063, max|grad|=6.210e-07
  Validacioni NLL po kolu: 19.20450779
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.01, početak=17, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.00427799, max|grad|=6.636e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.01, početak=11, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.00427799, max|grad|=5.585e-07
  Validacioni NLL po kolu: 18.63249038
KANDIDAT: lag=1, lambda=0.01, validacioni NLL=18.91849908
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.001, početak=17, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Prihvaćeno: gubitak=9.23514801, max|grad|=7.336e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.001, početak=11, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=9.14235700, max|grad|=8.443e-07
  Validacioni NLL po kolu: 24.44212392
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.001, početak=17, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Prihvaćeno: gubitak=10.08306935, max|grad|=8.669e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.001, početak=11, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Prihvaćeno: gubitak=10.17490676, max|grad|=7.871e-07
  Validacioni NLL po kolu: 24.09549961
KANDIDAT: lag=3, lambda=0.001, validacioni NLL=24.26881176
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.01, početak=17, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=10.63363648, max|grad|=6.187e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.01, početak=11, trening=1039
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Prihvaćeno: gubitak=10.68720879, max|grad|=9.842e-07
  Validacioni NLL po kolu: 24.47304668
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.01, početak=17, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Optimizacija geometrije: 1100 iteracija
  Optimizacija geometrije: 1200 iteracija
  Optimizacija geometrije: 1300 iteracija
  Optimizacija geometrije: 1400 iteracija
  Optimizacija geometrije: 1500 iteracija
  Prihvaćeno: gubitak=11.40409011, max|grad|=6.186e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.01, početak=11, trening=1212
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Optimizacija geometrije: 300 iteracija
  Optimizacija geometrije: 400 iteracija
  Optimizacija geometrije: 500 iteracija
  Optimizacija geometrije: 600 iteracija
  Optimizacija geometrije: 700 iteracija
  Optimizacija geometrije: 800 iteracija
  Optimizacija geometrije: 900 iteracija
  Optimizacija geometrije: 1000 iteracija
  Prihvaćeno: gubitak=11.46516677, max|grad|=9.168e-07
  Validacioni NLL po kolu: 24.05786118
KANDIDAT: lag=3, lambda=0.01, validacioni NLL=24.26545393

IZABRANO: lag=1, lambda=0.01, validacioni NLL=18.91849908
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=1, lambda=0.01, početak=17, trening=1385
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.16270237, max|grad|=8.574e-07
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=1, lambda=0.01, početak=11, trening=1385
  Optimizacija geometrije: 100 iteracija
  Optimizacija geometrije: 200 iteracija
  Prihvaćeno: gubitak=16.16270237, max|grad|=6.343e-07
ODVOJENI TEST: 347 kola, NLL po kolu=17.84446058
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=1, lambda=0.01, početak=17, trening=1732
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.33869329, max|grad|=9.542e-07
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=1, lambda=0.01, početak=11, trening=1732
  Optimizacija geometrije: 100 iteracija
  Prihvaćeno: gubitak=16.33869329, max|grad|=9.788e-07
ZAVRŠNO UČENJE: svih 1732 prelaza iz 1733 kola.



============================================================
ZAVRŠNI REZULTATI QUANTUM POGADJANJE V2
============================================================
NLL koristi prirodni logaritam.
Niži NLL označava bolji rezultat na tom skupu.
Modelna verovatnoća pripada naučenom modelu.
Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.

CSV: loto7_4710_k86.csv
NEXT: 03 06 08 10 14 26 29
Validacioni NLL: 16.91420333
Odvojeni test NLL: 16.79439308
Modelna verovatnoća NEXT: 1.23349262802e-06
Normalizaciona konstanta P(broj jedinica=7): 0.140019184552
Zbir uslovnih uključnih verovatnoća: 7.000000000000
Promena uslovnih uključnih verovatnoća kroz kontekst: 2.166324e-01
Najduži uzastopni niz: 1
Raspon kombinacije: 26
Izbor NEXT: tačan maksimum uslovne raspodele.

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 01 05 07 13 18 25 34
Validacioni NLL: 17.19381402
Odvojeni test NLL: 17.18068961
Modelna verovatnoća NEXT: 9.84986242863e-07
Normalizaciona konstanta P(broj jedinica=7): 0.141181119043
Zbir uslovnih uključnih verovatnoća: 7.000000000000
Promena uslovnih uključnih verovatnoća kroz kontekst: 2.581027e-01
Najduži uzastopni niz: 1
Raspon kombinacije: 33
Izbor NEXT: tačan maksimum uslovne raspodele.

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 08 13 25 28 31 34 39
Validacioni NLL: 18.91849908
Odvojeni test NLL: 17.84446058
Modelna verovatnoća NEXT: 6.35502371102e-06
Normalizaciona konstanta P(broj jedinica=7): 0.131674644946
Zbir uslovnih uključnih verovatnoća: 7.000000000000
Promena uslovnih uključnih verovatnoća kroz kontekst: 8.494228e-01
Najduži uzastopni niz: 1
Raspon kombinacije: 31
Izbor NEXT: tačan maksimum uslovne raspodele.

Učenje koristi lokalnu numeričku optimizaciju.
Globalni optimum parametara nije dokazan.
Zavisnosti izlaza ograničene su konstrukcijom protokola
i uslovom da kombinacija sadrži tačno sedam brojeva.
Simulacija ne dokazuje računarsku kvantnu prednost.
"""



"""
V2 - učiće se položaji Aliceinog „točkića” i Bobove merne baze. 
Verovatnoće će se računati iz kvantnih amplituda tog protokola, a zatim usloviti na tačno sedam različitih brojeva. 
Hronološka validacija ostaje zajednički okvir za proveru modela. 

V2 u glavnom računanju koristi spregnuti singlet, Aliceinu poruku i Bobovo merenje. 
Uče se uglovi kodiranja i merne baze. 
NEXT nastaje iz Bornovih verovatnoća, uslovljenih na sedam različitih brojeva.

Samoprovere stvarnog V2 fajla sada su izvršene i prošle su. 
U sačuvanom ispisu našao sam konkretan problem: za jedan kandidat trening gubitak pada na oko 14,8, dok validacioni NLL raste na oko 20,6. 
To pokazuje jako preprilagođavanje. 
"""
