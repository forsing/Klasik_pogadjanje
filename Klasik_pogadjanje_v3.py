#!/usr/bin/env python3



"""
V3: learned EARAC codebook, exact conditional seven-number distribution.
Known historical bits are encoded. Next-draw likelihood trains the codebook.
The communication protocol itself does not provide future information.
Every query uses a freshly prepared singlet. All Born probabilities are exact.
No random sampling, row shuffling, classifiers or cosmetic filters.
Only previous-draw bits enter a sample; all chronological transitions train.
Regularization controls codebook changes from its initial angular setting.
The initial setting is an optimization starting point, not an assumed draw law.
"""



import os

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import csv
import hashlib
import math
import sys
import traceback
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.optimize import minimize


NUMBER_COUNT = 39
DRAW_SIZE = 7

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
        alpha = np.sqrt(
            (1.0 + selected[:, 2]) / 2.0
        )
        beta = (
            selected[:, 0] + 1j * selected[:, 1]
        ) / (2.0 * alpha)

        states[north, 0] = alpha
        states[north, 1] = beta

    if np.any(south):
        selected = flat[south]
        beta = np.sqrt(
            (1.0 - selected[:, 2]) / 2.0
        )
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
    q = np.asarray(
        probability_one,
        dtype=np.float64,
    )
    r = np.asarray(
        probability_zero,
        dtype=np.float64,
    )

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
                conditional_marginals.sum(axis=1)
                - draw_size
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
        -np.sum(
            np.log(selected_probability),
            axis=1,
        )
        + log_z
    )

    require(
        np.all(np.isfinite(nll))
        and np.min(nll) >= -1e-7,
        "Neispravan gubitak kompletne kombinacije.",
    )

    return nll, selected_probability


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
            if not row or all(
                not item.strip() for item in row
            ):
                blank_rows += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: "
                "potrebno je 7 kolona.",
            )

            try:
                values = [
                    int(item.strip()) for item in row
                ]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: "
                    "neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == DRAW_SIZE,
                f"{path.name}, red {line_number}: "
                "ponovljen broj.",
            )

            require(
                all(
                    1 <= value <= NUMBER_COUNT
                    for value in values
                ),
                f"{path.name}, red {line_number}: "
                "broj van opsega 1–39.",
            )

            rows.append(
                sorted(value - 1 for value in values)
            )

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, "
        f"pročitano {len(rows)}.",
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


def verify_full_count_distribution(
    q,
    r,
    draw_size,
    log_z,
):
    number_count = len(q)

    distribution = np.zeros(
        number_count + 1,
        dtype=np.float64,
    )
    distribution[0] = 1.0

    for number in range(number_count):
        following = (
            r[number] * distribution
        )
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
            float(distribution[draw_size])
            - expected_z
        ) <= 1e-10 * expected_z,
        "Dve konstrukcije normalizacione konstante "
        "nisu saglasne.",
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
        required >= 0
        and required <= len(interior),
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
        float(
            np.sum(np.log(selected_probability))
        )
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
        first_bit = int(
            latest_bits[sample, 0]
        )

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
                    IDENTITY
                    + alice_sign * alice_operator
                ) / 2.0

                message = (
                    first_bit ^ alice_outcome
                )

                for bob_outcome in (0, 1):
                    bob_sign = (
                        -1.0 if bob_outcome == 0 else 1.0
                    )

                    bob_projector = (
                        IDENTITY
                        + bob_sign * bob_operator
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
                        "Projektorska verovatnoća ima "
                        "imaginarni ostatak.",
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
                    decoded_one
                    - probability_one[sample, query]
                ) < 1e-12,
                "Singlet amplitude i nezavisni projektorski "
                "račun nisu saglasni.",
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


def protocol(delta, latest, derivatives=False):
    n, m = latest.shape
    g = m // 3

    triples = latest.reshape(n, g, 3)

    codes = (
        4 * triples[:, :, 0]
        + 2 * triples[:, :, 1]
        + triples[:, :, 2]
    )

    angles = delta[
        np.arange(g)[None, :],
        codes,
    ]

    theta = (
        math.acos(1.0 / math.sqrt(3.0))
        + angles[:, :, 0]
    )
    phi = (
        math.pi / 4.0
        + angles[:, :, 1]
    )

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    corrected = np.stack(
        (st * cp, st * sp, ct),
        axis=-1,
    )

    alice_axis = (
        (1 - 2 * triples[:, :, 0])[:, :, None]
        * corrected
    )

    alice_states = (
        spinors_from_bloch(alice_axis),
        spinors_from_bloch(-alice_axis),
    )

    bob_axes = np.eye(3)

    bob_states = (
        spinors_from_bloch(-bob_axes),
        spinors_from_bloch(bob_axes),
    )

    one = np.zeros((n, g, 3))
    zero = np.zeros_like(one)

    for a in (0, 1):
        message = (
            triples[:, :, 0] ^ a
        )[:, :, None]

        aa = np.conj(alice_states[a])

        for b in (0, 1):
            bb = np.conj(bob_states[b])

            amplitude = (
                aa[:, :, 0, None]
                * bb[None, None, :, 1]
                - aa[:, :, 1, None]
                * bb[None, None, :, 0]
            ) / math.sqrt(2.0)

            probability = (
                np.abs(amplitude) ** 2
            )

            decoded = message ^ b

            one += probability * decoded
            zero += probability * (1 - decoded)

    total = one + zero

    require(
        np.max(np.abs(total - 1)) < 1e-10,
        "EARAC verovatnoće nisu normalizovane.",
    )

    one = one / total
    zero = zero / total

    require(
        np.max(
            np.abs(
                np.sum(
                    (zero - one) ** 2,
                    axis=2,
                ) - 1
            )
        ) < 1e-8,
        "Narušena je kvantna sferna granica.",
    )

    jac = None

    if derivatives:
        dt = np.stack(
            (ct * cp, ct * sp, -st),
            axis=-1,
        )
        dp = np.stack(
            (
                -st * sp,
                st * cp,
                np.zeros_like(st),
            ),
            axis=-1,
        )

        jac = np.stack(
            (dt, dp),
            axis=-2,
        )

    return (
        one.reshape(n, m),
        zero.reshape(n, m),
        codes,
        corrected,
        jac,
    )


def loss_gradient(flat, latest, targets, ridge):
    n, m = latest.shape
    g = m // 3

    delta = flat.reshape(g, 8, 2)

    q, r, codes, _, jac = protocol(
        delta,
        latest,
        True,
    )

    log_z, dz, _ = count_partition(
        q,
        r,
        targets.shape[1],
        True,
    )

    nll, chosen = set_nll(
        q,
        r,
        targets,
        log_z,
    )

    observed = np.zeros_like(q)

    observed[
        np.arange(n)[:, None],
        targets,
    ] = 1

    dq = (
        dz
        + (1 - 2 * observed) / chosen
    ) / n

    dr = (
        -0.5 * dq.reshape(n, g, 3)
    )

    da = np.einsum(
        "ngc,ngac->nga",
        dr,
        jac,
        optimize=True,
    )

    gradient = np.zeros_like(delta)

    np.add.at(
        gradient,
        (
            np.tile(np.arange(g), n),
            codes.ravel(),
        ),
        da.reshape(-1, 2),
    )

    value = float(
        np.mean(nll)
        + ridge * np.sum(delta * delta)
    )

    gradient += (
        2 * ridge * delta
    )

    require(
        np.isfinite(value)
        and np.all(np.isfinite(gradient)),
        "Neispravan gubitak ili izvod.",
    )

    return value, gradient.ravel()


def fit(latest, targets, ridge, label):
    shape = (
        latest.shape[1] // 3,
        8,
        2,
    )

    accepted = []

    for start in (0, 1):
        if start == 0:
            initial = np.zeros(
                np.prod(shape)
            )
        else:
            initial = (
                0.005
                * np.sin(
                    np.arange(np.prod(shape))
                    + 0.7
                )
            )

        result = None

        for _ in range(2):
            result = minimize(
                loss_gradient,
                initial,
                args=(
                    latest,
                    targets,
                    ridge,
                ),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": 500,
                    "maxls": 60,
                    "ftol": 1e-13,
                    "gtol": 1e-6,
                },
            )

            initial = result.x.copy()

            if (
                result.success
                and np.max(np.abs(result.jac)) <= 5e-5
            ):
                break

        require(
            result.success
            and np.max(np.abs(result.jac)) <= 5e-5,
            f"{label}: optimizacija nije prošla, "
            f"{result.message}",
        )

        accepted.append(result)

    best = min(
        accepted,
        key=lambda item: item.fun,
    )

    say(
        f"{label}: lambda={ridge}, "
        f"gubitak={best.fun:.8f}, "
        f"max|grad|={np.max(np.abs(best.jac)):.3e}"
    )

    return best.x.reshape(shape)


def evaluate(delta, latest, targets, hits=False):
    total = 0.0

    histogram = np.zeros(
        targets.shape[1] + 1,
        dtype=np.int64,
    )

    for start in range(0, len(targets), 512):
        stop = min(
            start + 512,
            len(targets),
        )

        q, r = protocol(
            delta,
            latest[start:stop],
        )[:2]

        log_z = count_partition(
            q,
            r,
            targets.shape[1],
        )[0]

        nll = set_nll(
            q,
            r,
            targets[start:stop],
            log_z,
        )[0]

        total += float(nll.sum())

        if hits:
            for j in range(len(q)):
                prediction = exact_next(
                    q[j],
                    r[j],
                    targets.shape[1],
                )[0]

                matched = len(
                    set(prediction).intersection(
                        targets[start + j].tolist()
                    )
                )

                histogram[matched] += 1

    return (
        total / len(targets),
        histogram,
    )


def self_checks():
    bits = np.array(
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

    delta = np.zeros((1, 8, 2))

    delta[0, :, 0] = (
        0.04 * np.sin(np.arange(8))
    )
    delta[0, :, 1] = (
        0.05 * np.cos(np.arange(8))
    )

    q, r, _, corrected, _ = protocol(
        delta,
        bits,
    )

    verify_protocol_by_projectors(
        bits,
        corrected,
        np.eye(3)[None, :, :],
        q,
    )

    require(
        np.max(
            np.abs(
                q
                - (1 - corrected[:, 0, :]) / 2
            )
        ) < 1e-12,
        "Bornov račun nije saglasan sa korelacijama.",
    )

    original = np.zeros_like(delta)

    signs = (
        1 - 2 * bits
    ) / math.sqrt(3.0)

    original[0, :, 0] = (
        np.arccos(signs[:, 2])
        - math.acos(1 / math.sqrt(3.0))
    )

    original[0, :, 1] = (
        np.arctan2(
            signs[:, 1],
            signs[:, 0],
        )
        - math.pi / 4
    )

    oq, oz = protocol(
        original,
        bits,
    )[:2]

    success = np.where(
        bits == 1,
        oq,
        oz,
    )

    require(
        np.max(
            np.abs(
                success
                - (1 + 1 / math.sqrt(3.0)) / 2
            )
        ) < 1e-12,
        "Izvorni trobitni EARAC nije prošao proveru.",
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

    flat = (
        0.01
        * np.sin(np.arange(2 * 8 * 2))
    )

    _, analytic = loss_gradient(
        flat,
        latest,
        targets,
        0.03,
    )

    numerical = np.empty_like(flat)

    for j in range(len(flat)):
        plus = flat.copy()
        minus = flat.copy()

        plus[j] += 1e-6
        minus[j] -= 1e-6

        numerical[j] = (
            loss_gradient(
                plus,
                latest,
                targets,
                0.03,
            )[0]
            - loss_gradient(
                minus,
                latest,
                targets,
                0.03,
            )[0]
        ) / 2e-6

    error = float(
        np.max(
            np.abs(analytic - numerical)
            / (
                1
                + np.abs(analytic)
                + np.abs(numerical)
            )
        )
    )

    require(
        error < 2e-5,
        "Izvod funkcije gubitka nije prošao proveru.",
    )

    q, r = protocol(
        flat.reshape(2, 8, 2),
        latest[:1],
    )[:2]

    cases = (
        (
            q[0],
            r[0],
            2,
        ),
        (
            np.array(
                [1.0, 0.0, 0.2, 0.4, 0.7, 0.6]
            ),
            np.array(
                [0.0, 1.0, 0.8, 0.6, 0.3, 0.4]
            ),
            2,
        ),
        (
            np.linspace(0.15, 0.30, 9),
            1 - np.linspace(0.15, 0.30, 9),
            7,
        ),
    )

    for cq, cr, k in cases:
        sets = list(
            combinations(
                range(len(cq)),
                k,
            )
        )

        values = []

        for selected in sets:
            probabilities = cr.copy()

            idx = np.asarray(
                selected,
                dtype=np.int64,
            )

            probabilities[idx] = cq[idx]

            values.append(
                float(np.prod(probabilities))
            )

        values = np.asarray(values)
        direct_z = float(values.sum())

        log_z = float(
            count_partition(
                cq[None, :],
                cr[None, :],
                k,
            )[0][0]
        )

        require(
            abs(
                math.exp(log_z) - direct_z
            ) < 1e-12,
            "Normalizacija nije prošla potpunu proveru.",
        )

        best, log_p, _ = exact_next(
            cq,
            cr,
            k,
        )

        require(
            abs(
                math.exp(log_p)
                - float(values.max() / direct_z)
            ) < 1e-12
            and abs(
                math.exp(log_p)
                - float(
                    values[sets.index(best)]
                    / direct_z
                )
            ) < 1e-12,
            "Maksimum nije prošao potpunu proveru.",
        )

    say(
        "Samoprovere prošle: izvorni EARAC, amplitude, "
        "poruka, merenja,",
        "sferna granica, izvodi, normalizacija, "
        "granični slučajevi i maksimum.",
        f"Relativna greška izvoda: {error:.3e}",
    )


def process(path, draws, occupancy, fingerprint):
    latest = occupancy[:-1]
    targets = draws[1:]

    n = len(targets)

    a = int(0.6 * n)
    b = int(0.7 * n)
    c = int(0.8 * n)

    require(
        50 < a < b < c < n,
        "Nema dovoljno podataka za hronološku podelu.",
    )

    scores = []

    for ridge in (0.03, 0.3, 3.0):
        total = 0.0
        count = 0

        for fold, (end, stop) in enumerate(
            ((a, b), (b, c)),
            1,
        ):
            model = fit(
                latest[:end],
                targets[:end],
                ridge,
                f"{path.name} / validacija {fold}",
            )

            value = evaluate(
                model,
                latest[end:stop],
                targets[end:stop],
            )[0]

            total += (
                (stop - end) * value
            )
            count += stop - end

            say(
                f"  Validacioni NLL: {value:.8f}"
            )

        score = total / count
        scores.append((score, ridge))

        say(
            f"KANDIDAT: lambda={ridge}, "
            f"zbirni validacioni NLL={score:.8f}"
        )

    validation_nll, ridge = min(
        scores,
        key=lambda item: item[0],
    )

    test_model = fit(
        latest[:c],
        targets[:c],
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_model,
        latest[c:],
        targets[c:],
        hits=True,
    )

    say(
        f"ODVOJENI TEST: NLL={test_nll:.8f}, "
        f"kola={n - c}"
    )
    say(
        "Pogodaka 0–7: "
        + " ".join(
            str(int(v))
            for v in histogram
        )
    )

    final = fit(
        latest,
        targets,
        ridge,
        f"{path.name} / završno učenje svih {n} prelaza",
    )

    q, r = protocol(
        final,
        occupancy[-1:],
    )[:2]

    best, log_p, log_z = exact_next(
        q[0],
        r[0],
        DRAW_SIZE,
    )

    _, _, marginal = count_partition(
        q,
        r,
        DRAW_SIZE,
        True,
    )

    old_q, old_r = protocol(
        final,
        latest[-8:],
    )[:2]

    old_marginal = count_partition(
        old_q,
        old_r,
        DRAW_SIZE,
        True,
    )[2]

    sensitivity = float(
        np.max(
            np.abs(
                old_marginal - marginal
            )
        )
    )

    repeats = int(
        np.count_nonzero(
            np.all(
                draws
                == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    run = longest_consecutive_run(best)

    warnings_list = []

    if run >= 5:
        warnings_list.append(
            f"Niz od {run} uzastopnih brojeva "
            "zahteva pregled modela."
        )

    if repeats:
        warnings_list.append(
            f"NEXT se pojavljuje {repeats} puta u istoriji."
        )

    if sensitivity < 1e-8:
        warnings_list.append(
            "Nije utvrđen uticaj promene istorijskog "
            "ulaza na proverenim kontekstima."
        )

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "next": tuple(v + 1 for v in best),
        "validation": validation_nll,
        "test": test_nll,
        "histogram": histogram,
        "test_count": n - c,
        "mean_hits": float(
            histogram @ np.arange(8)
            / (n - c)
        ),
        "probability": math.exp(log_p),
        "normalizer": math.exp(log_z),
        "sensitivity": sensitivity,
        "warnings": warnings_list,
    }


def main():
    np.seterr(
        over="raise",
        invalid="raise",
        divide="raise",
    )

    say(
        "QUANTUM POGADJANJE V3 — Loto 7/39",
        "Singlet amplitude, jedna Aliceina poruka "
        "i Bobov izbor merenja.",
        "Uči se 8 kodiranja po trobitnoj grupi: "
        "ukupno 208 parametara.",
        "Ulaz jednog prelaza je prethodno kolo. "
        "Učenje koristi sve prelaze.",
        "Lokalni klasični simulator. "
        "Bez slučajnog uzorkovanja.",
    )

    datasets = []

    for path, expected in FILES:
        draws, occupancy, fingerprint = load_draws(
            path,
            expected,
        )

        datasets.append(
            (
                path,
                draws,
                occupancy,
                fingerprint,
            )
        )

    require(
        len(
            {
                p.resolve()
                for p, _, _, _ in datasets
            }
        ) == 3,
        "Putanje se ponavljaju.",
    )

    require(
        len(
            {
                fp
                for _, _, _, fp in datasets
            }
        ) == 3,
        "CSV fajlovi imaju isti sadržaj.",
    )

    self_checks()

    results = [
        process(*dataset)
        for dataset in datasets
    ]

    for left in range(3):
        for right in range(left + 1, 3):
            if (
                results[left]["next"]
                == results[right]["next"]
            ):
                msg = (
                    "Ista NEXT kombinacija: "
                    f"{results[left]['name']} i "
                    f"{results[right]['name']}. "
                    "Proveriti model; rezultat nije promenjen."
                )

                results[left]["warnings"].append(msg)
                results[right]["warnings"].append(msg)

    say(
        "",
        "ZAVRŠNI REZULTATI V3",
        "Modelna verovatnoća nije potvrđena verovatnoća "
        "stvarnog izvlačenja.",
    )

    for result in results:
        say(
            "",
            f"CSV: {result['name']}",
            "NEXT: "
            + " ".join(
                f"{v:02d}"
                for v in result["next"]
            ),
            f"Validacioni NLL: {result['validation']:.8f}",
            f"Odvojeni test NLL: {result['test']:.8f}",
            "Prosečno pogodaka na odvojenom testu: "
            f"{result['mean_hits']:.6f}",
            "Test pogodaka 0–7: "
            + " ".join(
                str(int(v))
                for v in result["histogram"]
            ),
            "Modelna verovatnoća NEXT: "
            f"{result['probability']:.12g}",
            "Normalizaciona konstanta P(broj jedinica=7): "
            f"{result['normalizer']:.12g}",
            "Promena uslovnih uključnih verovatnoća "
            "kroz kontekst: "
            f"{result['sensitivity']:.6e}",
        )

        for message in result["warnings"]:
            say(
                "PROVERA MODELA: " + message
            )

    say(
        "",
        "NEXT je tačan maksimum naučene uslovne raspodele.",
        "Globalni optimum parametara i prediktivna "
        "prednost nisu dokazani.",
    )


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            np.seterr(
                over="raise",
                invalid="raise",
                divide="raise",
            )
            self_checks()
        else:
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
        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM POGADJANJE V3 — Loto 7/39
Singlet amplitude, jedna Aliceina poruka i Bobov izbor merenja.
Uči se 8 kodiranja po trobitnoj grupi: ukupno 208 parametara.
Ulaz jednog prelaza je prethodno kolo. Učenje koristi sve prelaze.
Lokalni klasični simulator. Bez slučajnog uzorkovanja.

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
Samoprovere prošle: izvorni EARAC, amplitude, poruka, merenja,
sferna granica, izvodi, normalizacija, granični slučajevi i maksimum.
Relativna greška izvoda: 2.443e-10
loto7_4710_k86.csv / validacija 1: lambda=0.03, gubitak=16.53222400, max|grad|=6.388e-07
  Validacioni NLL: 16.55795461
loto7_4710_k86.csv / validacija 2: lambda=0.03, gubitak=16.53448700, max|grad|=2.148e-07
  Validacioni NLL: 16.55531139
KANDIDAT: lambda=0.03, zbirni validacioni NLL=16.55663300
loto7_4710_k86.csv / validacija 1: lambda=0.3, gubitak=16.54380538, max|grad|=5.314e-07
  Validacioni NLL: 16.54948810
loto7_4710_k86.csv / validacija 2: lambda=0.3, gubitak=16.54443919, max|grad|=4.392e-07
  Validacioni NLL: 16.54902920
KANDIDAT: lambda=0.3, zbirni validacioni NLL=16.54925865
loto7_4710_k86.csv / validacija 1: lambda=3.0, gubitak=16.54794177, max|grad|=1.567e-07
  Validacioni NLL: 16.54855867
loto7_4710_k86.csv / validacija 2: lambda=3.0, gubitak=16.54802994, max|grad|=1.456e-07
  Validacioni NLL: 16.54852303
KANDIDAT: lambda=3.0, zbirni validacioni NLL=16.54854085
loto7_4710_k86.csv / odvojeni test: lambda=3.0, gubitak=16.54805777, max|grad|=1.424e-07
ODVOJENI TEST: NLL=16.54833354, kola=942
Pogodaka 0–7: 197 402 261 70 12 0 0 0
loto7_4710_k86.csv / završno učenje svih 4709 prelaza: lambda=3.0, gubitak=16.54810729, max|grad|=1.378e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lambda=0.03, gubitak=16.51708439, max|grad|=8.797e-07
  Validacioni NLL: 16.56792260
loto7_4710_k86_loto_2977.csv / validacija 2: lambda=0.03, gubitak=16.52129420, max|grad|=4.814e-07
  Validacioni NLL: 16.56726893
KANDIDAT: lambda=0.03, zbirni validacioni NLL=16.56759632
loto7_4710_k86_loto_2977.csv / validacija 1: lambda=0.3, gubitak=16.53901800, max|grad|=1.671e-07
  Validacioni NLL: 16.54962460
loto7_4710_k86_loto_2977.csv / validacija 2: lambda=0.3, gubitak=16.54014568, max|grad|=5.324e-07
  Validacioni NLL: 16.55102703
KANDIDAT: lambda=0.3, zbirni validacioni NLL=16.55032464
loto7_4710_k86_loto_2977.csv / validacija 1: lambda=3.0, gubitak=16.54724232, max|grad|=4.412e-08
  Validacioni NLL: 16.54834570
loto7_4710_k86_loto_2977.csv / validacija 2: lambda=3.0, gubitak=16.54740191, max|grad|=3.612e-08
  Validacioni NLL: 16.54863903
KANDIDAT: lambda=3.0, zbirni validacioni NLL=16.54849212
loto7_4710_k86_loto_2977.csv / odvojeni test: lambda=3.0, gubitak=16.54757466, max|grad|=1.560e-07
ODVOJENI TEST: NLL=16.54796810, kola=596
Pogodaka 0–7: 121 229 167 67 12 0 0 0
loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza: lambda=3.0, gubitak=16.54768281, max|grad|=1.539e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lambda=0.03, gubitak=16.49295821, max|grad|=3.739e-07
  Validacioni NLL: 16.59395767
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lambda=0.03, gubitak=16.50355372, max|grad|=3.326e-07
  Validacioni NLL: 16.56594921
KANDIDAT: lambda=0.03, zbirni validacioni NLL=16.57995344
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lambda=0.3, gubitak=16.53072488, max|grad|=4.379e-07
  Validacioni NLL: 16.55634875
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lambda=0.3, gubitak=16.53423344, max|grad|=9.676e-08
  Validacioni NLL: 16.54935367
KANDIDAT: lambda=0.3, zbirni validacioni NLL=16.55285121
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lambda=3.0, gubitak=16.54595505, max|grad|=1.023e-07
  Validacioni NLL: 16.54899099
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lambda=3.0, gubitak=16.54648409, max|grad|=1.111e-07
  Validacioni NLL: 16.54803839
KANDIDAT: lambda=3.0, zbirni validacioni NLL=16.54851469
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lambda=3.0, gubitak=16.54670840, max|grad|=1.290e-07
ODVOJENI TEST: NLL=16.54960932, kola=347
Pogodaka 0–7: 76 152 90 22 6 1 0 0
loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza: lambda=3.0, gubitak=16.54725290, max|grad|=5.591e-08

ZAVRŠNI REZULTATI V3
Modelna verovatnoća nije potvrđena verovatnoća stvarnog izvlačenja.

CSV: loto7_4710_k86.csv
NEXT: 03 08 10 16 21 23 39
Validacioni NLL: 16.54854085
Odvojeni test NLL: 16.54833354
Prosečno pogodaka na odvojenom testu: 1.254777
Test pogodaka 0–7: 197 402 261 70 12 0 0 0
Modelna verovatnoća NEXT: 6.76454811946e-08
Normalizaciona konstanta P(broj jedinica=7): 0.14533688696
Promena uslovnih uključnih verovatnoća kroz kontekst: 2.194204e-03

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 10 13 16 21 22 26 39
Validacioni NLL: 16.54849212
Odvojeni test NLL: 16.54796810
Prosečno pogodaka na odvojenom testu: 1.362416
Test pogodaka 0–7: 121 229 167 67 12 0 0 0
Modelna verovatnoća NEXT: 6.83260474807e-08
Normalizaciona konstanta P(broj jedinica=7): 0.145336559323
Promena uslovnih uključnih verovatnoća kroz kontekst: 2.245957e-03

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 04 07 11 18 23 25 29
Validacioni NLL: 16.54851469
Odvojeni test NLL: 16.54960932
Prosečno pogodaka na odvojenom testu: 1.230548
Test pogodaka 0–7: 76 152 90 22 6 1 0 0
Modelna verovatnoća NEXT: 6.85983288084e-08
Normalizaciona konstanta P(broj jedinica=7): 0.145336166559
Promena uslovnih uključnih verovatnoća kroz kontekst: 2.394760e-03

NEXT je tačan maksimum naučene uslovne raspodele.
Globalni optimum parametara i prediktivna prednost nisu dokazani.
"""



"""
V3 sa zajedničkim Aliceinim kontrolama, što smanjuje broj parametara uz isti singlet, poruku i Bobovo merenje.
Prošli su matematički testovi, validacija, odvojeni test i završno učenje.

Konkretna izmena: umesto hiljada parametara iz V2, uči se 208 parametara EARAC kodiranja. 
Singlet, Aliceina poruka i Bobovo merenje ostaju glavni račun. 

CSV	    Dobijeni NEXT	        Prosečno pogodaka na testu
4710	03 08 10 16 21 23 39	1,255
2977	10 13 16 21 22 26 39	1,362
1733	04 07 11 18 23 25 29	1,231

Prediktivna prednost nije potvrđena. 
"""
