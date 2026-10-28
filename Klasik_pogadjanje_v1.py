#!/usr/bin/env python3



"""
QUANTUM BOX V1 — Loto 7/39
Python 3.11
Dependencies: numpy, scipy

Adaptation of the three-bit EARAC/QRAC construction.

Quantum part:
    Encode each historical draw as 39 occupancy bits.
    Divide these into thirteen groups of three bits.
    Prepare the equivalent three-bit QRAC amplitudes.
    Compute exact Pauli expectations from the amplitudes.
    Include two-qubit Pauli correlations between cyclically
    adjacent groups in the product encoding.

    Self-check the equivalent EARAC primitive using:
        a two-qubit singlet;
        Alice's projectors;
        Bob's selected Pauli measurement;
        Alice's classical message;
        Bob's decoded bit.

    Each expectation represents a separate preparation/readout.
    No simultaneous measurement of incompatible observables
    on a single copy is assumed.

Classical adaptation:
    Learn two readout branches and their mixture weight.
    Fit the normalized distribution of complete seven-number sets.
    Do not fit independent binary classifiers.

For branch j:

    h_j(x) = CAP * tanh(readout_j(x) / CAP)

    P_j(S | x) = exp(sum(h_ji(x), i in S)) / Z_j(x)

    Z_j(x) = elementary symmetric polynomial e_7(exp(h_j(x)))

Final distribution:

    P(S | x) = pi_0(x) P_0(S | x) + pi_1(x) P_1(S | x)

This is an adaptation for temporal distribution regression.
The original communication protocol does not predict future draws.

No random initialization, sampling, shuffling or random splits.
No rounding, duplicate repair or cosmetic combination filters.
No quantum hardware or Qiskit circuit execution.

Chronological validation:
    Train first 60%; validate next 10%.
    Train first 70%; validate next 10%.
    Select parameters from validation NLL only.
    Train first 80%; evaluate final 20% once.
    Refit on ALL historical transitions.

Test evaluation is one-step conditional evaluation:
    each target uses only draws preceding that target.

NEXT:
    Exhaustive search over all legal seven-number combinations.
    Independently sum all probabilities during that search.
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
from itertools import combinations, islice
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp



FILES = (
    (
        Path("/data/loto7_4710_k86.csv"),
        4710,
    ),
    (
        Path("/data/loto7_4710_k86_loto_2977.csv"),
        2977,
    ),
    (
        Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
        1733,
    ),
)



NUMBER_COUNT = 39
DRAW_SIZE = 7
GROUP_COUNT = 13
BRANCH_COUNT = 2

FEATURE_DIM = 16
CAP = 8.0

CANDIDATES = (
    (1, 0.03),
    (1, 0.30),
    (3, 0.03),
    (3, 0.30),
)

INITIAL_ORDERS = (17, 11)

OPTIMIZER_STEPS = 600
OPTIMIZER_RESTARTS = 3
GRADIENT_LIMIT = 5e-4

MAP_BATCH = 8192
NORMALIZATION_TOLERANCE = 2e-7

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


def make_quantum_lookup():
    states = np.empty((8, 2), dtype=np.complex128)
    expected_bloch = np.empty((8, 3), dtype=np.float64)

    for code in range(8):
        bits = np.array(
            [
                (code >> 2) & 1,
                (code >> 1) & 1,
                code & 1,
            ],
            dtype=np.int64,
        )

        r = (1.0 - 2.0 * bits) / math.sqrt(3.0)

        alpha = math.sqrt((1.0 + r[2]) / 2.0)
        beta = (
            (r[0] + 1j * r[1])
            / math.sqrt(2.0 * (1.0 + r[2]))
        )

        states[code] = (alpha, beta)
        expected_bloch[code] = r

    # Compute readouts from amplitudes, not from the bit labels.
    bloch = np.empty((8, 3), dtype=np.float64)

    for code, state in enumerate(states):
        for axis in range(3):
            bloch[code, axis] = float(
                np.vdot(
                    state,
                    PAULI[axis] @ state,
                ).real
            )

    require(
        np.max(np.abs(bloch - expected_bloch)) < 1e-12,
        "Amplitudno kodiranje ne daje očekivane Pauli vrednosti.",
    )
    require(
        np.max(
            np.abs(
                np.sum(np.abs(states) ** 2, axis=1)
                - 1.0
            )
        ) < 1e-12,
        "Kodirana kvantna stanja nisu normalizovana.",
    )

    return states, bloch


def quantum_draw_features(draws, bloch_lookup):
    occupancy = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.int64,
    )
    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1

    triples = occupancy.reshape(
        len(draws),
        GROUP_COUNT,
        3,
    )

    codes = (
        4 * triples[:, :, 0]
        + 2 * triples[:, :, 1]
        + triples[:, :, 2]
    )

    r = bloch_lookup[codes]

    single_readouts = r.reshape(len(draws), -1)

    adjacent = np.roll(r, -1, axis=1)

    pair_readouts = np.einsum(
        "ngi,ngj->ngij",
        r,
        adjacent,
        optimize=True,
    ).reshape(len(draws), -1)

    features = np.concatenate(
        (single_readouts, pair_readouts),
        axis=1,
    )

    require(
        features.shape == (len(draws), 156),
        "Neispravan oblik kvantnih karakteristika.",
    )
    require(
        np.all(np.isfinite(features)),
        "Kvantne karakteristike nisu konačne.",
    )

    return features


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

            # Sort within the draw. Keep chronological row order.
            rows.append(sorted(value - 1 for value in values))

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, pročitano {len(rows)}.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    require(
        draws.shape == (expected_rows, DRAW_SIZE),
        "Neispravan oblik istorije.",
    )

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank_rows}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, fingerprint


def make_context(draw_features, target_indices, lookback):
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    require(
        target_indices.ndim == 1,
        "Neispravni ciljni indeksi.",
    )
    require(
        np.all(
            (target_indices >= 1)
            & (target_indices <= len(draw_features))
        ),
        "Ciljni indeks van opsega.",
    )

    feature_count = draw_features.shape[1]
    block_size = feature_count + 1

    result = np.zeros(
        (len(target_indices), lookback * block_size),
        dtype=np.float64,
    )

    for lag in range(1, lookback + 1):
        previous = target_indices - lag
        valid = np.flatnonzero(previous >= 0)
        offset = (lag - 1) * block_size

        if valid.size:
            result[
                valid,
                offset:offset + feature_count,
            ] = draw_features[previous[valid]]

            result[
                valid,
                offset + feature_count,
            ] = 1.0

    return result


def fit_feature_map(raw_x):
    mean = raw_x.mean(axis=0)
    centered = raw_x - mean

    covariance = (
        centered.T @ centered / len(centered)
    )

    values, vectors = np.linalg.eigh(covariance)

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    threshold = max(float(values[0]), 1.0) * 1e-12

    available = int(
        np.count_nonzero(values > threshold)
    )
    dimension = min(FEATURE_DIM, available)

    require(
        dimension > 0,
        "Istorijski kvantni ulazi nemaju promenljive karakteristike.",
    )

    values = values[:dimension]
    vectors = vectors[:, :dimension].copy()

    for column in range(dimension):
        pivot = int(
            np.argmax(np.abs(vectors[:, column]))
        )
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0

    projection = (
        vectors / np.sqrt(values)[None, :]
    )

    return mean, projection


def transform_features(raw_x, mean, projection):
    transformed = (raw_x - mean) @ projection

    return np.column_stack(
        (
            np.ones(len(transformed)),
            transformed,
        )
    )


def fixed_size_partition(fields, draw_size, with_marginals=False):
    """
    Exact normalizer for a fixed-size subset distribution.

    Z = e_k(exp(fields))

    The reverse recurrence returns exact inclusion marginals.
    No Bernoulli classifier is fitted.
    """
    fields = np.asarray(fields, dtype=np.float64)

    require(
        fields.ndim == 2
        and fields.shape[0] > 0
        and 1 <= draw_size <= fields.shape[1],
        "Neispravan oblik raspodele fiksne veličine.",
    )
    require(
        np.all(np.isfinite(fields)),
        "Polja raspodele nisu konačna.",
    )

    sample_count, number_count = fields.shape
    weights = np.exp(fields)

    forward = np.zeros(
        (sample_count, draw_size + 1),
        dtype=np.float64,
    )
    forward[:, 0] = 1.0

    cache = [forward] if with_marginals else None

    for number in range(number_count):
        following = forward.copy()
        following[:, 1:] += (
            weights[:, number, None]
            * forward[:, :-1]
        )

        forward = following

        if with_marginals:
            cache.append(forward)

    z = forward[:, draw_size]

    require(
        np.all(np.isfinite(z)) and np.all(z > 0),
        "Normalizaciona konstanta nije pozitivna i konačna.",
    )

    log_z = np.log(z)

    if not with_marginals:
        return log_z, None

    reverse = np.zeros_like(forward)
    reverse[:, draw_size] = 1.0 / z

    marginals = np.zeros_like(fields)

    for number in range(number_count - 1, -1, -1):
        previous = cache[number]

        marginals[:, number] = (
            weights[:, number]
            * np.sum(
                reverse[:, 1:] * previous[:, :-1],
                axis=1,
            )
        )

        previous_reverse = reverse.copy()
        previous_reverse[:, :-1] += (
            weights[:, number, None]
            * reverse[:, 1:]
        )

        reverse = previous_reverse

    require(
        np.all(np.isfinite(marginals)),
        "Uključne verovatnoće nisu konačne.",
    )
    require(
        np.min(marginals) >= -1e-9
        and np.max(marginals) <= 1.0 + 1e-9,
        "Uključna verovatnoća je van opsega [0,1].",
    )
    require(
        np.max(
            np.abs(
                marginals.sum(axis=1) - draw_size
            )
        ) < 1e-8,
        "Zbir uključnih verovatnoća nije jednak veličini kombinacije.",
    )

    return log_z, marginals


def pack_parameters(theta, gate):
    return np.concatenate(
        (theta.ravel(), gate.ravel())
    )


def unpack_parameters(parameters, feature_count, number_count):
    size = BRANCH_COUNT * feature_count * number_count

    theta = parameters[:size].reshape(
        BRANCH_COUNT,
        feature_count,
        number_count,
    )
    gate = parameters[size:].reshape(feature_count)

    return theta, gate


def readout_components(theta, gate, x):
    raw = np.einsum(
        "nf,hfm->nhm",
        x,
        theta,
        optimize=True,
    )

    t = np.tanh(raw / CAP)
    fields = CAP * t
    link_derivative = 1.0 - t * t

    gate_value = x @ gate

    log_pi = np.column_stack(
        (
            -np.logaddexp(0.0, gate_value),
            -np.logaddexp(0.0, -gate_value),
        )
    )

    require(
        np.all(np.isfinite(fields))
        and np.all(np.isfinite(log_pi)),
        "Očitavanje modela nije konačno.",
    )

    return fields, link_derivative, log_pi


def probability_terms(
    theta,
    gate,
    x,
    targets,
    draw_size,
    with_marginals=False,
):
    fields, link_derivative, log_pi = readout_components(
        theta,
        gate,
        x,
    )

    sample_count = len(targets)
    rows = np.arange(sample_count)[:, None]

    component_log_probability = np.empty(
        (sample_count, BRANCH_COUNT),
        dtype=np.float64,
    )
    marginals = (
        np.empty_like(fields)
        if with_marginals
        else None
    )

    for branch in range(BRANCH_COUNT):
        branch_fields = fields[:, branch, :]

        log_z, branch_marginals = fixed_size_partition(
            branch_fields,
            draw_size,
            with_marginals=with_marginals,
        )

        component_log_probability[:, branch] = (
            np.sum(
                branch_fields[rows, targets],
                axis=1,
            )
            - log_z
        )

        if with_marginals:
            marginals[:, branch, :] = branch_marginals

    joint = log_pi + component_log_probability
    log_probability = logsumexp(joint, axis=1)

    responsibilities = np.exp(
        joint - log_probability[:, None]
    )

    require(
        np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(responsibilities)),
        "Neispravna verovatnoća kompletne kombinacije.",
    )
    require(
        float(np.max(log_probability)) <= 1e-7,
        "Verovatnoća kompletne kombinacije prelazi 1.",
    )

    return (
        log_probability,
        responsibilities,
        marginals,
        link_derivative,
        log_pi,
    )


def objective(
    parameters,
    x,
    targets,
    ridge,
    number_count,
    draw_size,
):
    theta, gate = unpack_parameters(
        parameters,
        x.shape[1],
        number_count,
    )

    (
        log_probability,
        responsibilities,
        marginals,
        link_derivative,
        log_pi,
    ) = probability_terms(
        theta,
        gate,
        x,
        targets,
        draw_size,
        with_marginals=True,
    )

    sample_count = len(targets)

    observed = np.zeros(
        (sample_count, number_count),
        dtype=np.float64,
    )
    observed[
        np.arange(sample_count)[:, None],
        targets,
    ] = 1.0

    gradient_fields = (
        responsibilities[:, :, None]
        * (marginals - observed[:, None, :])
        * link_derivative
    )

    gradient_theta = np.einsum(
        "nf,nhm->hfm",
        x,
        gradient_fields,
        optimize=True,
    ) / sample_count

    pi_one = np.exp(log_pi[:, 1])

    gradient_gate = (
        x.T
        @ (pi_one - responsibilities[:, 1])
        / sample_count
    )

    penalty_mask = np.ones(x.shape[1], dtype=np.float64)
    penalty_mask[0] = 0.1

    penalty = ridge * (
        np.sum(
            theta * theta
            * penalty_mask[None, :, None]
        )
        + np.sum(
            gate * gate * penalty_mask
        )
    )

    gradient_theta += (
        2.0
        * ridge
        * theta
        * penalty_mask[None, :, None]
    )
    gradient_gate += (
        2.0
        * ridge
        * gate
        * penalty_mask
    )

    loss = (
        -float(np.mean(log_probability))
        + float(penalty)
    )

    gradient = pack_parameters(
        gradient_theta,
        gradient_gate,
    )

    require(
        np.isfinite(loss)
        and np.all(np.isfinite(gradient)),
        "Funkcija gubitka ili njen izvod nisu konačni.",
    )

    return loss, gradient


def initial_parameters(feature_count, number_count, phase_order):
    theta = np.zeros(
        (BRANCH_COUNT, feature_count, number_count),
        dtype=np.float64,
    )

    labels = (
        phase_order * np.arange(number_count) + 11
    ) % number_count

    angle = 2.0 * np.pi * labels / number_count

    theta[0, 0, :] = 0.03 * np.cos(angle)
    theta[1, 0, :] = -theta[0, 0, :]

    if feature_count > 1:
        theta[0, 1, :] = 0.06 * np.sin(angle)
        theta[1, 1, :] = -theta[0, 1, :]

    if feature_count > 2:
        theta[0, 2, :] = 0.04 * np.cos(2.0 * angle)
        theta[1, 2, :] = -theta[0, 2, :]

    gate = np.zeros(feature_count, dtype=np.float64)

    return pack_parameters(theta, gate)


@dataclass
class Model:
    theta: np.ndarray
    gate: np.ndarray
    mean: np.ndarray
    projection: np.ndarray
    lookback: int
    ridge: float
    train_loss: float
    gradient_norm: float
    initial_order: int

    def transform(self, raw_x):
        return transform_features(
            raw_x,
            self.mean,
            self.projection,
        )

    def components(self, raw_row, draw_size=DRAW_SIZE):
        x = self.transform(
            np.asarray(raw_row).reshape(1, -1)
        )

        fields, _, log_pi = readout_components(
            self.theta,
            self.gate,
            x,
        )

        fields = fields[0]
        log_pi = log_pi[0]

        log_z = np.array(
            [
                fixed_size_partition(
                    fields[branch:branch + 1],
                    draw_size,
                )[0][0]
                for branch in range(BRANCH_COUNT)
            ],
            dtype=np.float64,
        )

        return fields, log_pi, log_z


def fit_model(raw_x, targets, lookback, ridge, label):
    mean, projection = fit_feature_map(raw_x)
    x = transform_features(raw_x, mean, projection)

    fitted = []

    for phase_order in INITIAL_ORDERS:
        say(
            f"{label}: lag={lookback}, lambda={ridge}, "
            f"početak={phase_order}, trening={len(targets)}"
        )

        parameters = initial_parameters(
            x.shape[1],
            NUMBER_COUNT,
            phase_order,
        )

        iteration = 0

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 100 == 0:
                say(
                    f"  Optimizacija: {iteration} iteracija"
                )

        def function(current):
            return objective(
                current,
                x,
                targets,
                ridge,
                NUMBER_COUNT,
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

            theta, gate = unpack_parameters(
                parameters,
                x.shape[1],
                NUMBER_COUNT,
            )

            fitted.append(
                Model(
                    theta=theta.copy(),
                    gate=gate.copy(),
                    mean=mean.copy(),
                    projection=projection.copy(),
                    lookback=lookback,
                    ridge=ridge,
                    train_loss=float(loss),
                    gradient_norm=gradient_norm,
                    initial_order=phase_order,
                )
            )

            say(
                f"  Prihvaćen lokalni minimum: gubitak={loss:.8f}, "
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


def evaluate_nll(model, raw_x, targets):
    total = 0.0

    for start in range(0, len(targets), 512):
        stop = min(start + 512, len(targets))

        log_probability = probability_terms(
            model.theta,
            model.gate,
            model.transform(raw_x[start:stop]),
            targets[start:stop],
            DRAW_SIZE,
        )[0]

        total -= float(np.sum(log_probability))

    return total / len(targets)


def set_log_probabilities(fields, log_pi, log_z, sets):
    branch_zero = (
        log_pi[0]
        + np.sum(fields[0][sets], axis=1)
        - log_z[0]
    )
    branch_one = (
        log_pi[1]
        + np.sum(fields[1][sets], axis=1)
        - log_z[1]
    )

    return np.logaddexp(branch_zero, branch_one)


def exhaustive_maximum(
    fields,
    log_pi,
    log_z,
    draw_size,
    verbose=True,
    label="NEXT",
):
    number_count = fields.shape[1]
    expected = math.comb(number_count, draw_size)

    iterator = combinations(
        range(number_count),
        draw_size,
    )

    visited = 0
    best = None
    best_log_probability = -math.inf
    total_log_probability = -math.inf
    previous_progress = 0

    while True:
        block = list(islice(iterator, MAP_BATCH))

        if not block:
            break

        sets = np.asarray(block, dtype=np.int64)

        scores = set_log_probabilities(
            fields,
            log_pi,
            log_z,
            sets,
        )

        require(
            np.all(np.isfinite(scores)),
            "Završna pretraga je dala neispravnu log-verovatnoću.",
        )

        local_index = int(np.argmax(scores))
        local_score = float(scores[local_index])

        if local_score > best_log_probability:
            best_log_probability = local_score
            best = tuple(
                int(value)
                for value in sets[local_index]
            )

        total_log_probability = float(
            np.logaddexp(
                total_log_probability,
                logsumexp(scores),
            )
        )

        visited += len(sets)
        progress = visited // 2_000_000

        if verbose and progress > previous_progress:
            say(
                f"{label}: pregledano {visited:,}/{expected:,}"
            )
            previous_progress = progress

    require(
        visited == expected,
        "Završna pretraga nije potpuna.",
    )
    require(
        best is not None
        and np.isfinite(best_log_probability)
        and best_log_probability <= 1e-7,
        "Neispravan maksimum raspodele.",
    )

    probability_sum = math.exp(
        total_log_probability
    )

    require(
        abs(probability_sum - 1.0)
        <= NORMALIZATION_TOLERANCE,
        f"Normalizacija nije prošla: zbir={probability_sum:.12g}.",
    )

    return (
        best,
        best_log_probability,
        probability_sum,
        visited,
    )


def check_earac(states, bloch_lookup):
    singlet = np.array(
        [0.0, 1.0, -1.0, 0.0],
        dtype=np.complex128,
    ) / math.sqrt(2.0)

    expected_success = (
        0.5 * (1.0 + 1.0 / math.sqrt(3.0))
    )

    for code in range(8):
        bits = np.array(
            [
                (code >> 2) & 1,
                (code >> 1) & 1,
                code & 1,
            ],
            dtype=np.int64,
        )

        signs = 1.0 - 2.0 * bits
        alice_axis = (
            signs * signs[0] / math.sqrt(3.0)
        )

        alice_operator = np.einsum(
            "a,aij->ij",
            alice_axis,
            PAULI,
        )

        for axis in range(3):
            decoded_expectation = 0.0
            success = 0.0
            total = 0.0

            for alice_outcome in (0, 1):
                alice_sign = (
                    1.0 if alice_outcome == 0 else -1.0
                )
                alice_projector = (
                    IDENTITY + alice_sign * alice_operator
                ) / 2.0

                message = int(bits[0]) ^ alice_outcome

                for bob_sign in (-1.0, 1.0):
                    bob_projector = (
                        IDENTITY + bob_sign * PAULI[axis]
                    ) / 2.0

                    joint_projector = np.kron(
                        alice_projector,
                        bob_projector,
                    )

                    probability_complex = np.vdot(
                        singlet,
                        joint_projector @ singlet,
                    )

                    require(
                        abs(probability_complex.imag) < 1e-12,
                        "EARAC verovatnoća ima imaginarni ostatak.",
                    )

                    probability = float(
                        probability_complex.real
                    )

                    require(
                        probability >= -1e-12,
                        "EARAC verovatnoća je negativna.",
                    )

                    bob_outcome = (
                        0 if bob_sign < 0 else 1
                    )
                    guess = message ^ bob_outcome

                    total += probability
                    decoded_expectation += (
                        probability
                        * (1.0 if guess == 0 else -1.0)
                    )

                    if guess == int(bits[axis]):
                        success += probability

            require(
                abs(total - 1.0) < 1e-12,
                "EARAC zajedničke verovatnoće nisu normalizovane.",
            )
            require(
                abs(success - expected_success) < 1e-12,
                "EARAC rekonstrukcija istorijskog bita nije prošla.",
            )
            require(
                abs(
                    decoded_expectation
                    - bloch_lookup[code, axis]
                ) < 1e-12,
                "EARAC i ekvivalentno amplitudno kodiranje nisu saglasni.",
            )

    pair_state = np.kron(states[1], states[5])
    pair_operator = np.kron(PAULI[1], PAULI[2])

    actual = np.vdot(
        pair_state,
        pair_operator @ pair_state,
    )

    expected = (
        bloch_lookup[1, 1]
        * bloch_lookup[5, 2]
    )

    require(
        abs(actual - expected) < 1e-12,
        "Provera dvokubitnog Pauli očitavanja nije prošla.",
    )


def self_checks(states, bloch_lookup):
    say(
        "",
        "Pokrećem matematičke i numeričke samoprovere."
    )

    check_earac(states, bloch_lookup)

    number_count = 9
    draw_size = 3

    x = np.array(
        [
            [1.0, 0.2, -0.4],
            [1.0, -0.3, 0.1],
            [1.0, 0.7, 0.5],
            [1.0, -0.6, -0.2],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [0, 2, 5],
            [1, 4, 8],
            [0, 3, 7],
            [2, 6, 8],
        ],
        dtype=np.int64,
    )

    parameters = initial_parameters(
        x.shape[1],
        number_count,
        phase_order=2,
    )

    # Include a nonzero mixture readout in the gradient check.
    parameters[-x.shape[1]:] = (0.03, -0.04, 0.02)

    _, analytic = objective(
        parameters,
        x,
        targets,
        0.03,
        number_count,
        draw_size,
    )

    epsilon = 1e-6
    numerical = np.empty_like(parameters)

    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        f_plus = objective(
            plus,
            x,
            targets,
            0.03,
            number_count,
            draw_size,
        )[0]

        f_minus = objective(
            minus,
            x,
            targets,
            0.03,
            number_count,
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

    theta, gate = unpack_parameters(
        parameters,
        x.shape[1],
        number_count,
    )

    all_sets = np.asarray(
        list(
            combinations(
                range(number_count),
                draw_size,
            )
        ),
        dtype=np.int64,
    )

    repeated_x = np.repeat(
        x[:1],
        len(all_sets),
        axis=0,
    )

    log_probability = probability_terms(
        theta,
        gate,
        repeated_x,
        all_sets,
        draw_size,
    )[0]

    probability_sum = float(
        np.exp(logsumexp(log_probability))
    )

    require(
        abs(probability_sum - 1.0) < 1e-10,
        "Potpuna mala provera normalizacije nije prošla.",
    )

    fields, _, log_pi = readout_components(
        theta,
        gate,
        x[:1],
    )

    fields = fields[0]
    log_pi = log_pi[0]

    log_z = np.array(
        [
            fixed_size_partition(
                fields[branch:branch + 1],
                draw_size,
            )[0][0]
            for branch in range(BRANCH_COUNT)
        ]
    )

    for branch in range(BRANCH_COUNT):
        direct_log_z = logsumexp(
            np.sum(fields[branch][all_sets], axis=1)
        )

        require(
            abs(direct_log_z - log_z[branch]) < 1e-11,
            "Rekurzija normalizacije nije saglasna sa nabrajanjem.",
        )

    best, best_log_probability, _, _ = exhaustive_maximum(
        fields,
        log_pi,
        log_z,
        draw_size,
        verbose=False,
    )

    matches = np.flatnonzero(
        np.all(
            all_sets == np.asarray(best),
            axis=1,
        )
    )

    require(
        len(matches) == 1
        and abs(
            best_log_probability
            - float(np.max(log_probability))
        ) < 1e-10
        and abs(
            best_log_probability
            - float(log_probability[matches[0]])
        ) < 1e-10,
        "Provera maksimuma nije prošla.",
    )

    # Check the seven-number normalizer independently.
    seven_sets = np.asarray(
        list(combinations(range(9), 7)),
        dtype=np.int64,
    )
    reference_fields = (
        0.2 * np.cos(np.arange(9))
    )

    recursive_log_z = fixed_size_partition(
        reference_fields[None, :],
        7,
    )[0][0]

    direct_log_z = logsumexp(
        np.sum(reference_fields[seven_sets], axis=1)
    )

    require(
        abs(recursive_log_z - direct_log_z) < 1e-11,
        "Provera normalizacije za sedam brojeva nije prošla.",
    )

    toy_draws = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.int64,
    )

    toy_features = quantum_draw_features(
        toy_draws,
        bloch_lookup,
    )

    context = make_context(
        toy_features,
        [1],
        lookback=3,
    )

    block_size = toy_features.shape[1] + 1

    require(
        context[0, block_size - 1] == 1.0
        and np.all(context[0, block_size:] == 0.0)
        and np.allclose(
            context[0, :block_size - 1],
            toy_features[0],
        ),
        "Provera početnog istorijskog prelaza nije prošla.",
    )

    say(
        "Samoprovere prošle: amplitude, EARAC, Pauli očitavanje,",
        "izvod, normalizacija, maksimum i početni prelaz.",
        f"Relativna greška izvoda: {gradient_error:.3e}",
        "EARAC provera odnosi se na rekonstrukciju poznatih bitova.",
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


def check_context_sensitivity(model, raw, raw_next, draws):
    probes = np.unique(
        draws[-min(16, len(draws)):],
        axis=0,
    )

    fields, log_pi, log_z = model.components(raw_next)

    reference = set_log_probabilities(
        fields,
        log_pi,
        log_z,
        probes,
    )

    sensitivity = 0.0

    for previous in raw[-8:]:
        old_fields, old_log_pi, old_log_z = model.components(
            previous
        )

        old_probability = set_log_probabilities(
            old_fields,
            old_log_pi,
            old_log_z,
            probes,
        )

        sensitivity = max(
            sensitivity,
            float(
                np.max(
                    np.abs(reference - old_probability)
                )
            ),
        )

    return sensitivity


def process_file(path, draws, fingerprint, bloch_lookup):
    draw_features = quantum_draw_features(
        draws,
        bloch_lookup,
    )

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
                draw_features,
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
        raw = contexts[lookback]
        total_nll = 0.0
        total_count = 0

        for fold_number, (
            train_end,
            validation_end,
        ) in enumerate(folds, start=1):
            model = fit_model(
                raw[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                label=f"{path.name} / validacija {fold_number}",
            )

            count = validation_end - train_end

            nll = evaluate_nll(
                model,
                raw[train_end:validation_end],
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

    raw = contexts[selected_lookback]

    say(
        "",
        f"IZABRANO: lag={selected_lookback}, "
        f"lambda={selected_ridge}, "
        f"validacioni NLL={validation_nll:.8f}",
    )

    test_model = fit_model(
        raw[:train_80],
        targets[:train_80],
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        raw[train_80:],
        targets[train_80:],
    )

    say(
        f"ODVOJENI TEST: {sample_count - train_80} kola, "
        f"NLL po kolu={test_nll:.8f}",
        "Test se ne koristi za promenu parametara.",
    )

    final_model = fit_model(
        raw,
        targets,
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / završno učenje",
    )

    say(
        f"ZAVRŠNO UČENJE: svih {sample_count} prelaza "
        f"iz {len(draws)} kola."
    )

    raw_next = make_context(
        draw_features,
        [len(draws)],
        selected_lookback,
    )[0]

    sensitivity = check_context_sensitivity(
        final_model,
        raw,
        raw_next,
        draws,
    )

    fields, log_pi, log_z = final_model.components(
        raw_next
    )

    best, log_probability, probability_sum, visited = (
        exhaustive_maximum(
            fields,
            log_pi,
            log_z,
            DRAW_SIZE,
            verbose=True,
            label=path.name,
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
            "Na proverenim kombinacijama nije utvrđen numerički "
            "uticaj promene konteksta. Proveriti korišćenje istorije."
        )

    # Each branch is invariant to a common offset of its fields.
    centered_fields = (
        fields - fields.mean(axis=1, keepdims=True)
    )

    branch_difference = float(
        np.max(
            np.abs(
                centered_fields[0] - centered_fields[1]
            )
        )
    )

    if branch_difference < 1e-8:
        warnings.append(
            "Dve grane imaju numerički istu raspodelu za NEXT. "
            "Model se u ovom kontekstu svodi na jednu granu."
        )

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "combination": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "model_probability": math.exp(log_probability),
        "probability_sum": probability_sum,
        "visited": visited,
        "sensitivity": sensitivity,
        "branch_weights": np.exp(log_pi),
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
        "Broj oznaka nije saglasan sa trobitnim kodiranjem.",
    )

    say(
        "QUANTUM BOX V1 — Loto 7/39",
        "Trobitni EARAC i ekvivalentno amplitudno kodiranje istorije.",
        "Dve naučene grane očitavanja.",
        "Regresija raspodele kompletnih kombinacija.",
        "Lokalni klasični simulator; bez Qiskit kola.",
        "Bez slučajnog generisanja i bez klasifikatora.",
        "Učenje i završna pretraga mogu potrajati.",
    )

    states, bloch_lookup = make_quantum_lookup()

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
        draws, fingerprint = load_draws(
            path,
            expected_rows,
        )
        datasets.append((path, draws, fingerprint))

    require(
        len(
            {
                fingerprint
                for _, _, fingerprint in datasets
            }
        ) == 3,
        "Dva CSV-a imaju identičan sadržaj.",
    )

    self_checks(states, bloch_lookup)

    results = []

    for path, draws, fingerprint in datasets:
        results.append(
            process_file(
                path,
                draws,
                fingerprint,
                bloch_lookup,
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
        "ZAVRŠNI REZULTATI QUANTUM BOX V1",
        "============================================================",
        "Modelna verovatnoća pripada naučenoj raspodeli.",
        "Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.",
        "NLL koristi prirodni logaritam.",
        "Niži NLL označava bolji rezultat na tom skupu.",
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
            f"Pregledano kombinacija: {result['visited']:,}",
            "Zbir svih verovatnoća: "
            f"{result['probability_sum']:.12f}",
            "Težine grana za NEXT: "
            f"{result['branch_weights'][0]:.6f} "
            f"{result['branch_weights'][1]:.6f}",
            "Promena log-verovatnoće u proveri konteksta: "
            f"{result['sensitivity']:.6e}",
            f"Najduži uzastopni niz: {result['run_length']}",
            f"Raspon kombinacije: {result['span']}",
        )

        for warning in result["warnings"]:
            say(
                f"PROVERA MODELA: {warning}"
            )

    say(
        "",
        "Završna pretraga je potpuna za svaki prihvaćeni model.",
        "Učenje je lokalna numerička optimizacija.",
        "Globalni optimum parametara nije dokazan.",
        "Ograničenje: dve grane predstavljaju ograničenu porodicu raspodela.",
        "Simulacija ne predstavlja dokaz računarske kvantne prednosti.",
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
QUANTUM BOX V1 — Loto 7/39
Trobitni EARAC i ekvivalentno amplitudno kodiranje istorije.
Dve naučene grane očitavanja.
Regresija raspodele kompletnih kombinacija.
Lokalni klasični simulator; bez Qiskit kola.
Bez slučajnog generisanja i bez klasifikatora.
Učenje i završna pretraga mogu potrajati.

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
Samoprovere prošle: amplitude, EARAC, Pauli očitavanje,
izvod, normalizacija, maksimum i početni prelaz.
Relativna greška izvoda: 7.025e-10
EARAC provera odnosi se na rekonstrukciju poznatih bitova.
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početak=17, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.44947281, max|grad|=3.514e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početak=11, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.44791013, max|grad|=7.489e-07
  Validacioni NLL po kolu: 16.58341054
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početak=17, trening=3296
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46466181, max|grad|=9.019e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početak=11, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.46466181, max|grad|=8.037e-07
  Validacioni NLL po kolu: 16.59444755
KANDIDAT: lag=1, lambda=0.03, validacioni NLL=16.58892904
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početak=17, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.53244738, max|grad|=3.219e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početak=11, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.53244738, max|grad|=9.195e-07
  Validacioni NLL po kolu: 16.54894761
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početak=17, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.53469087, max|grad|=7.334e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početak=11, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.53469087, max|grad|=6.780e-07
  Validacioni NLL po kolu: 16.54961946
KANDIDAT: lag=1, lambda=0.3, validacioni NLL=16.54928353
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početak=17, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.44521331, max|grad|=7.638e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početak=11, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.44600207, max|grad|=8.591e-07
  Validacioni NLL po kolu: 16.60271537
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početak=17, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.46286956, max|grad|=8.477e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početak=11, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.46266744, max|grad|=9.306e-07
  Validacioni NLL po kolu: 16.56646355
KANDIDAT: lag=3, lambda=0.03, validacioni NLL=16.58458946
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početak=17, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.53143151, max|grad|=3.494e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početak=11, trening=2825
  Prihvaćen lokalni minimum: gubitak=16.53143151, max|grad|=8.846e-07
  Validacioni NLL po kolu: 16.55147383
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početak=17, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.53390544, max|grad|=8.260e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početak=11, trening=3296
  Prihvaćen lokalni minimum: gubitak=16.53390544, max|grad|=7.605e-07
  Validacioni NLL po kolu: 16.54664076
KANDIDAT: lag=3, lambda=0.3, validacioni NLL=16.54905730

IZABRANO: lag=3, lambda=0.3, validacioni NLL=16.54905730
loto7_4710_k86.csv / odvojeni test: lag=3, lambda=0.3, početak=17, trening=3767
  Prihvaćen lokalni minimum: gubitak=16.53527247, max|grad|=5.921e-07
loto7_4710_k86.csv / odvojeni test: lag=3, lambda=0.3, početak=11, trening=3767
  Prihvaćen lokalni minimum: gubitak=16.53527247, max|grad|=5.409e-07
ODVOJENI TEST: 942 kola, NLL po kolu=16.54628173
Test se ne koristi za promenu parametara.
loto7_4710_k86.csv / završno učenje: lag=3, lambda=0.3, početak=17, trening=4709
  Prihvaćen lokalni minimum: gubitak=16.53681295, max|grad|=5.887e-07
loto7_4710_k86.csv / završno učenje: lag=3, lambda=0.3, početak=11, trening=4709
  Prihvaćen lokalni minimum: gubitak=16.53681295, max|grad|=8.426e-07
ZAVRŠNO UČENJE: svih 4709 prelaza iz 4710 kola.
loto7_4710_k86.csv: pregledano 2,007,040/15,380,937
loto7_4710_k86.csv: pregledano 4,005,888/15,380,937
loto7_4710_k86.csv: pregledano 6,004,736/15,380,937
loto7_4710_k86.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86.csv: pregledano 14,000,128/15,380,937
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početak=17, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.37926293, max|grad|=8.417e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početak=11, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.37895879, max|grad|=5.597e-07
  Validacioni NLL po kolu: 16.59866762
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početak=17, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.40560613, max|grad|=7.879e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početak=11, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.40326183, max|grad|=7.131e-07
  Validacioni NLL po kolu: 16.62334959
KANDIDAT: lag=1, lambda=0.03, validacioni NLL=16.61098786
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početak=17, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.52425213, max|grad|=9.360e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početak=11, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.52425213, max|grad|=8.353e-07
  Validacioni NLL po kolu: 16.54327311
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početak=17, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.52621815, max|grad|=6.385e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početak=11, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.52621815, max|grad|=5.692e-07
  Validacioni NLL po kolu: 16.55427405
KANDIDAT: lag=1, lambda=0.3, validacioni NLL=16.54876433
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početak=17, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.37880494, max|grad|=7.456e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početak=11, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.37880494, max|grad|=9.533e-07
  Validacioni NLL po kolu: 16.60092764
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početak=17, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.40389258, max|grad|=6.709e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početak=11, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.40235914, max|grad|=6.139e-07
  Validacioni NLL po kolu: 16.62540105
KANDIDAT: lag=3, lambda=0.03, validacioni NLL=16.61314378
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početak=17, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.52368446, max|grad|=6.384e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početak=11, trening=1785
  Prihvaćen lokalni minimum: gubitak=16.52368446, max|grad|=7.973e-07
  Validacioni NLL po kolu: 16.54395038
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početak=17, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.52617281, max|grad|=9.213e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početak=11, trening=2083
  Prihvaćen lokalni minimum: gubitak=16.52617281, max|grad|=8.244e-07
  Validacioni NLL po kolu: 16.55061809
KANDIDAT: lag=3, lambda=0.3, validacioni NLL=16.54727863

IZABRANO: lag=3, lambda=0.3, validacioni NLL=16.54727863
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=3, lambda=0.3, početak=17, trening=2380
  Prihvaćen lokalni minimum: gubitak=16.52827946, max|grad|=7.036e-07
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=3, lambda=0.3, početak=11, trening=2380
  Prihvaćen lokalni minimum: gubitak=16.52827946, max|grad|=6.074e-07
ODVOJENI TEST: 596 kola, NLL po kolu=16.54860176
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_2977.csv / završno učenje: lag=3, lambda=0.3, početak=17, trening=2976
  Prihvaćen lokalni minimum: gubitak=16.53256503, max|grad|=5.273e-07
loto7_4710_k86_loto_2977.csv / završno učenje: lag=3, lambda=0.3, početak=11, trening=2976
  Prihvaćen lokalni minimum: gubitak=16.53256503, max|grad|=9.319e-07
ZAVRŠNO UČENJE: svih 2976 prelaza iz 2977 kola.
loto7_4710_k86_loto_2977.csv: pregledano 2,007,040/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 4,005,888/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 6,004,736/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 14,000,128/15,380,937
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početak=17, trening=1039
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.22446627, max|grad|=7.508e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početak=11, trening=1039
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.22446627, max|grad|=9.450e-07
  Validacioni NLL po kolu: 16.71724077
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početak=17, trening=1212
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.28654804, max|grad|=6.050e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početak=11, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.28654804, max|grad|=6.078e-07
  Validacioni NLL po kolu: 16.70878641
KANDIDAT: lag=1, lambda=0.03, validacioni NLL=16.71301359
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početak=17, trening=1039
  Prihvaćen lokalni minimum: gubitak=16.50515491, max|grad|=7.559e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početak=11, trening=1039
  Prihvaćen lokalni minimum: gubitak=16.50515491, max|grad|=7.701e-07
  Validacioni NLL po kolu: 16.56561821
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početak=17, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.51238815, max|grad|=8.506e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početak=11, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.51238815, max|grad|=9.768e-07
  Validacioni NLL po kolu: 16.56254909
KANDIDAT: lag=1, lambda=0.3, validacioni NLL=16.56408365
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početak=17, trening=1039
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.20717661, max|grad|=5.532e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početak=11, trening=1039
  Prihvaćen lokalni minimum: gubitak=16.20717661, max|grad|=5.996e-07
  Validacioni NLL po kolu: 16.71171008
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početak=17, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.28874094, max|grad|=1.712e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početak=11, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.27781955, max|grad|=3.924e-07
  Validacioni NLL po kolu: 16.64402134
KANDIDAT: lag=3, lambda=0.03, validacioni NLL=16.67786571
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početak=17, trening=1039
  Prihvaćen lokalni minimum: gubitak=16.50277205, max|grad|=9.003e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početak=11, trening=1039
  Prihvaćen lokalni minimum: gubitak=16.50277205, max|grad|=7.579e-07
  Validacioni NLL po kolu: 16.56089357
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početak=17, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.51134359, max|grad|=7.563e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početak=11, trening=1212
  Prihvaćen lokalni minimum: gubitak=16.51134359, max|grad|=9.555e-07
  Validacioni NLL po kolu: 16.56257917
KANDIDAT: lag=3, lambda=0.3, validacioni NLL=16.56173637

IZABRANO: lag=3, lambda=0.3, validacioni NLL=16.56173637
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=3, lambda=0.3, početak=17, trening=1385
  Prihvaćen lokalni minimum: gubitak=16.51609907, max|grad|=7.300e-07
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=3, lambda=0.3, početak=11, trening=1385
  Prihvaćen lokalni minimum: gubitak=16.51609907, max|grad|=9.673e-07
ODVOJENI TEST: 347 kola, NLL po kolu=16.55435690
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=3, lambda=0.3, početak=17, trening=1732
  Prihvaćen lokalni minimum: gubitak=16.52226662, max|grad|=9.861e-07
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=3, lambda=0.3, početak=11, trening=1732
  Prihvaćen lokalni minimum: gubitak=16.52226662, max|grad|=7.829e-07
ZAVRŠNO UČENJE: svih 1732 prelaza iz 1733 kola.
loto7_4710_k86_loto_plus_1733.csv: pregledano 2,007,040/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 4,005,888/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 6,004,736/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 14,000,128/15,380,937



============================================================
ZAVRŠNI REZULTATI QUANTUM BOX V1
============================================================
Modelna verovatnoća pripada naučenoj raspodeli.
Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.
NLL koristi prirodni logaritam.
Niži NLL označava bolji rezultat na tom skupu.

CSV: loto7_4710_k86.csv
NEXT: 08 x 11 y 25 z 31
Validacioni NLL: 16.54905730
Odvojeni test NLL: 16.54628173
Modelna verovatnoća NEXT: 9.17612370504e-08
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Težine grana za NEXT: 0.500000 0.500000
Promena log-verovatnoće u proveri konteksta: 1.479822e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 23

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 06 x 09 y 23 z 34
Validacioni NLL: 16.54727863
Odvojeni test NLL: 16.54860176
Modelna verovatnoća NEXT: 9.03234815257e-08
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Težine grana za NEXT: 0.500000 0.500000
Promena log-verovatnoće u proveri konteksta: 2.064974e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 28

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 x 10 y 26 z 33
Validacioni NLL: 16.56173637
Odvojeni test NLL: 16.55435690
Modelna verovatnoća NEXT: 1.02125868104e-07
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Težine grana za NEXT: 0.499999 0.500001
Promena log-verovatnoće u proveri konteksta: 2.508414e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 31

Završna pretraga je potpuna za svaki prihvaćeni model.
Učenje je lokalna numerička optimizacija.
Globalni optimum parametara nije dokazan.
Ograničenje: dve grane predstavljaju ograničenu porodicu raspodela.
Simulacija ne predstavlja dokaz računarske kvantne prednosti.
"""



"""
Kvantno kodiranje poznate istorije i izbor načina njenog očitavanja. 
Entanglement-assisted random access codes — EARAC. 
Alice zna niz bitova. Bob želi da sazna jedan određeni bit. 
Uz zajedničko kvantno stanje i ograničenu komunikaciju može ga rekonstruisati uspešnije nego odgovarajućim klasičnim postupkom. 
Za osnovni slučaj sa dva bita uspeh je:
\[
p=\frac12\left(1+\frac1{\sqrt2}\right)\approx85{,}36\%.
\]
To je uspeh pri očitavanju već poznatih i kodiranih podataka, pod uslovima tog protokola. 

Za Loto: istorijska kola znam, dok narednu kombinaciju nema ko unapred da kodira. 
Zato se ne može preneti na loto predikciju.

Kodiranje više klasičnih bitova u kvantno stanje. 	
Prethodno kolo predstavim sa 39 oznaka prisustva brojeva, od kojih je sedam jedinica.
Geometrija amplituda i faza	Grupe istorijskih oznaka kodiram u mala kvantna stanja. 
QRAC postupci daju konkretne konstrukcije takvog kodiranja.
Različitim mernim operatorima izdvajam različite osobine iste kodirane istorije.
Proveravam da li kompaktnije predstavljanje pomaže učenju ili gubi korisnu istorijsku strukturu.

Konstrukcije za kodiranje dva ili tri bita u jedan qubit opisane su u radu o QRAC metodama. 
EARAC dodatno koristi zajedničku spregnutost i klasičnu poruku; ta dva postupka treba jasno razlikovati. 

1. Prethodna kola kodiram amplitudama, koristeći samo podatke dostupne pre ciljnog kola.
2. Učim parametre kodiranja i očitavanja kroz regresiju raspodele naredne kombinacije.
3. Dobijam zajedničku raspodelu nad skupovima od tačno sedam različitih brojeva.
4. NEXT određujem maksimumom te raspodele, bez slučajnog uzorkovanja.
Koraci učenja vremenskih prelaza i izgradnje raspodele.
Uspešno očitavanje jednog bita nije potvrda da sam dobio ispravnu zajedničku raspodelu sedam brojeva.

Kod koristi kvantno kodiranje prethodnih kola, naučeno očitavanje i raspodelu nad kombinacijama od sedam različitih brojeva. 
Hronološka validacija, odvojeni test i zasebnu obradu sva tri CSV-a. 

CSV učitavanje, hronološke podele, L-BFGS optimizaciju, validacioni NLL i potpunu NEXT pretragu. 

Sam EARAC protokol proverava se u samoproverama. 
On nije glavni prediktivni mehanizam završnog modela. 
Uspešna samoprovera protokola zato ne potvrđuje njegov doprinos loto predikciji.
Nova ulazna transformacija i drugi oblik raspodele. 
"""
