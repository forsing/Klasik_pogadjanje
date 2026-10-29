import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import traceback

from itertools import combinations, islice
from pathlib import Path

import numpy as np

from scipy.optimize import minimize
from scipy.special import expit, logsumexp


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)

NUMBER_COUNT = 39
DRAW_SIZE = 7
RIDGES = (0.03, 0.3, 3.0)
SEARCH_BATCH = 50000


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class PyramidModel:
    """
    Alice:
        Two previous occupancy vectors.
        Trainable leaf representations.
        Pairwise pyramidal compression.
        Probability of one binary communication bit.

    Bob:
        Two sets of query-dependent inclusion potentials.
        One set for each possible communication bit.

    Each bit-conditioned distribution contains exactly k selected labels.
    The final distribution sums both possible communication outcomes.
    """

    def __init__(self, m=39, k=7):
        require(1 <= k <= m, "Neispravne dimenzije modela.")

        self.m = m
        self.k = k

        self.levels = []
        width = m

        while width > 1:
            pairs = width // 2
            self.levels.append((width, pairs))
            width = pairs + width % 2

        self.node_count = sum(pairs for _, pairs in self.levels)
        self.size = 3 * m + 3 * self.node_count + 2 + 2 * m

    def initial(self):
        theta = np.zeros(self.size, dtype=np.float64)
        position = 0

        leaf = theta[position:position + 3 * self.m].reshape(3, self.m)
        labels = np.arange(1, self.m + 1, dtype=float)

        leaf[0] = 0.02 * np.sin(labels)
        leaf[1] = 0.20 * np.cos(labels * 0.71)
        leaf[2] = 0.15 * np.sin(labels * 0.53)

        position += 3 * self.m

        for _, pairs in self.levels:
            weights = theta[position:position + 3 * pairs].reshape(pairs, 3)
            weights[:, 0] = 0.65
            weights[:, 1] = 0.65
            weights[:, 2] = 0.01 * np.sin(np.arange(pairs) + 1)
            position += 3 * pairs

        theta[position:position + 2] = (0.0, 1.0)
        position += 2

        decoder = theta[position:].reshape(2, self.m)
        decoder[0] = 0.04 * np.sin(labels * 0.43)
        decoder[1] = -decoder[0]

        return theta

    def unpack(self, theta):
        require(theta.shape == (self.size,), "Pogrešna dužina parametara.")

        position = 0
        leaf = theta[position:position + 3 * self.m].reshape(3, self.m)
        position += 3 * self.m

        nodes = []

        for _, pairs in self.levels:
            nodes.append(
                theta[position:position + 3 * pairs].reshape(pairs, 3)
            )
            position += 3 * pairs

        gate = theta[position:position + 2]
        position += 2
        decoder = theta[position:].reshape(2, self.m)

        return leaf, nodes, gate, decoder

    def encode(self, theta, x):
        leaf, nodes, gate, _ = self.unpack(theta)

        require(
            x.ndim == 3 and x.shape[1:] == (2, self.m),
            "Pogrešan oblik istorijskog ulaza.",
        )

        z = (
            leaf[0][None, :]
            + x[:, 0, :] * leaf[1][None, :]
            + x[:, 1, :] * leaf[2][None, :]
        )

        current = np.tanh(z)
        leaf_output = current
        traces = []

        for (width, pairs), weights in zip(self.levels, nodes):
            left = current[:, :2 * pairs:2]
            right = current[:, 1:2 * pairs:2]

            paired = np.tanh(
                left * weights[:, 0]
                + right * weights[:, 1]
                + weights[:, 2]
            )

            traces.append((current, paired))

            if width % 2:
                current = np.concatenate(
                    (paired, current[:, -1:]),
                    axis=1,
                )
            else:
                current = paired

        root = current[:, 0]
        logits = gate[0] + gate[1] * root

        require(np.all(np.isfinite(logits)), "Nevažeća Alice poruka.")

        return logits, (leaf_output, traces, root)

    def encoder_gradient(self, theta, x, cache, dlogits):
        _, nodes, gate, _ = self.unpack(theta)
        leaf_output, traces, root = cache

        gate_gradient = np.array(
            (
                np.sum(dlogits),
                np.dot(dlogits, root),
            )
        )

        upstream = (dlogits * gate[1])[:, None]
        node_gradients = [None] * len(self.levels)

        for level in range(len(self.levels) - 1, -1, -1):
            width, pairs = self.levels[level]
            weights = nodes[level]
            previous, paired = traces[level]

            delta = upstream[:, :pairs] * (1.0 - paired * paired)

            left = previous[:, :2 * pairs:2]
            right = previous[:, 1:2 * pairs:2]

            node_gradients[level] = np.column_stack(
                (
                    np.sum(delta * left, axis=0),
                    np.sum(delta * right, axis=0),
                    np.sum(delta, axis=0),
                )
            )

            preceding = np.zeros_like(previous)
            preceding[:, :2 * pairs:2] = delta * weights[:, 0]
            preceding[:, 1:2 * pairs:2] = delta * weights[:, 1]

            if width % 2:
                preceding[:, -1] = upstream[:, -1]

            upstream = preceding

        delta = upstream * (1.0 - leaf_output * leaf_output)

        leaf_gradient = np.vstack(
            (
                np.sum(delta, axis=0),
                np.sum(delta * x[:, 0, :], axis=0),
                np.sum(delta * x[:, 1, :], axis=0),
            )
        )

        parts = [leaf_gradient.ravel()]
        parts.extend(value.ravel() for value in node_gradients)
        parts.append(gate_gradient)

        return np.concatenate(parts)


def subset_statistics(h, k):
    """
    Exact partition function and inclusion marginals for:
        P(S) proportional to exp(sum(h[i] for i in S))
        subject to len(S) == k.
    """

    m = len(h)

    forward = np.full((m + 1, k + 1), -np.inf)
    backward = np.full((m + 1, k + 1), -np.inf)

    forward[0, 0] = 0.0
    backward[m, 0] = 0.0

    for i in range(m):
        forward[i + 1, 0] = 0.0

        for count in range(1, min(k, i + 1) + 1):
            forward[i + 1, count] = np.logaddexp(
                forward[i, count],
                forward[i, count - 1] + h[i],
            )

    for i in range(m - 1, -1, -1):
        backward[i, 0] = 0.0

        for count in range(1, min(k, m - i) + 1):
            backward[i, count] = np.logaddexp(
                backward[i + 1, count],
                backward[i + 1, count - 1] + h[i],
            )

    logz = float(forward[m, k])
    require(np.isfinite(logz), "Nevažeća normalizacija dekodera.")

    inclusion = np.empty(m, dtype=float)

    for i in range(m):
        terms = (
            forward[i, :k]
            + backward[i + 1, k - 1::-1]
            + h[i]
        )
        inclusion[i] = math.exp(float(logsumexp(terms)) - logz)

    require(
        np.all(inclusion >= -1e-10)
        and np.all(inclusion <= 1.0 + 1e-10),
        "Marginalne verovatnoće van granica.",
    )

    require(
        abs(float(inclusion.sum()) - k) < 1e-8,
        "Zbir marginalnih verovatnoća nije sedam.",
    )

    return logz, inclusion


def distribution_terms(model, theta, x, y=None):
    logits, cache = model.encode(theta, x)
    _, _, _, decoder = model.unpack(theta)

    stats = [
        subset_statistics(decoder[bit], model.k)
        for bit in range(2)
    ]

    logz = np.array([item[0] for item in stats])
    marginals = np.stack([item[1] for item in stats])

    loggate = np.column_stack(
        (
            -np.logaddexp(0.0, logits),
            -np.logaddexp(0.0, -logits),
        )
    )

    if y is None:
        return logits, cache, decoder, logz, marginals, loggate

    component_logp = y @ decoder.T - logz[None, :]
    joint = component_logp + loggate
    logp = logsumexp(joint, axis=1)

    require(np.all(np.isfinite(logp)), "Nevažeća verovatnoća cilja.")
    require(np.all(logp <= 1e-8), "Verovatnoća cilja veća od jedan.")

    responsibilities = np.exp(joint - logp[:, None])

    return logits, cache, marginals, logp, responsibilities


def objective(theta, model, x, y, ridge):
    n = len(y)

    logits, cache, marginals, logp, responsibilities = distribution_terms(
        model, theta, x, y
    )

    dlogits = (expit(logits) - responsibilities[:, 1]) / n

    encoder_gradient = model.encoder_gradient(
        theta, x, cache, dlogits
    )

    decoder_gradient = (
        responsibilities.sum(axis=0)[:, None] * marginals
        - responsibilities.T @ y
    ) / n

    gradient = np.concatenate(
        (encoder_gradient, decoder_gradient.ravel())
    )

    penalty_weights = np.ones_like(theta)

    # Do not penalize the gate slope or pairwise averaging weights
    # toward zero. Penalize deviations from deterministic initialization.
    reference = model.initial()
    difference = theta - reference

    loss = (
        -float(logp.mean())
        + 0.5 * ridge * float(np.dot(
            penalty_weights * difference,
            difference,
        ))
    )

    gradient += ridge * penalty_weights * difference

    require(
        np.isfinite(loss) and np.all(np.isfinite(gradient)),
        "Nevažeća funkcija cilja ili gradijent.",
    )

    return loss, gradient


def fit(model, x, y, ridge, label):
    theta = model.initial()
    steps = 0

    def callback(value):
        nonlocal steps
        steps += 1

        if steps % 50 == 0:
            loss, gradient = objective(value, model, x, y, ridge)
            say(
                f"{label} | iteracija {steps}"
                f" | cilj {loss:.9f}"
                f" | gradijent {np.max(np.abs(gradient)):.3e}"
            )

    for attempt in range(3):
        result = minimize(
            objective,
            theta,
            args=(model, x, y, ridge),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": 700,
                "maxls": 60,
                "maxcor": 20,
                "ftol": 1e-13,
                "gtol": 1e-7,
            },
        )

        theta = result.x
        loss, gradient = objective(theta, model, x, y, ridge)
        maximum_gradient = float(np.max(np.abs(gradient)))

        if maximum_gradient <= 2e-5:
            say(
                f"{label} | završen"
                f" | cilj {loss:.9f}"
                f" | gradijent {maximum_gradient:.3e}"
            )
            return theta

        say(
            f"{label} | nastavak optimizacije {attempt + 1}"
            f" | {result.message}"
        )

    raise RuntimeError(
        f"{label}: optimizacija nije dostigla zahtevani gradijent."
    )


def nll(model, theta, x, y):
    _, _, _, logp, _ = distribution_terms(model, theta, x, y)
    return -float(logp.mean())


def contexts(occupancy, targets):
    m = occupancy.shape[1]
    center = DRAW_SIZE / m

    x = np.zeros((len(targets), 2, m), dtype=float)

    for lag in range(2):
        indices = targets - lag - 1
        valid = indices >= 0
        x[valid, lag, :] = occupancy[indices[valid]] - center

    return x


def load_csv(path):
    raw = path.read_bytes()
    rows = []
    empty = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not value.strip() for value in row):
                empty += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: očekujem sedam kolona.",
            )

            try:
                numbers = tuple(int(value.strip()) for value in row)
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == DRAW_SIZE
                and min(numbers) >= 1
                and max(numbers) <= NUMBER_COUNT,
                f"{path.name}, red {line_number}: neispravna sedmorka.",
            )

            rows.append(tuple(sorted(numbers)))

    require(len(rows) >= 100, f"{path.name}: nedovoljno istorije.")

    occupancy = np.zeros((len(rows), NUMBER_COUNT), dtype=float)

    for i, row in enumerate(rows):
        occupancy[i, np.asarray(row) - 1] = 1.0

    digest = hashlib.sha256(raw).hexdigest()

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(rows)} | praznih redova: {empty}",
        f"SHA256: {digest}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return rows, occupancy, digest


def exact_next(model, theta, context):
    """
    Exhaustive MAP search of the mixture over all legal combinations.
    No independent ranking approximation is used.
    """

    _, _, decoder, logz, _, loggate = distribution_terms(
        model, theta, context
    )

    require(len(context) == 1, "NEXT zahteva jedan istorijski kontekst.")

    offsets = loggate[0] - logz
    iterator = combinations(range(model.m), model.k)

    total = math.comb(model.m, model.k)
    completed = 0
    best_logp = -np.inf
    best = None

    while True:
        group = list(islice(iterator, SEARCH_BATCH))

        if not group:
            break

        candidates = np.asarray(group, dtype=np.int16)
        scores = np.stack(
            (
                decoder[0][candidates].sum(axis=1) + offsets[0],
                decoder[1][candidates].sum(axis=1) + offsets[1],
            ),
            axis=1,
        )

        logp = np.logaddexp(scores[:, 0], scores[:, 1])
        index = int(np.argmax(logp))
        value = float(logp[index])

        # Iteration is lexicographic. Exact ties retain the first tuple.
        if value > best_logp:
            best_logp = value
            best = tuple(int(v) + 1 for v in candidates[index])

        previous = completed
        completed += len(group)

        if completed // 1000000 != previous // 1000000:
            say(f"Tačna NEXT pretraga: {completed} / {total}")

    require(completed == total, "NEXT pretraga nije potpuna.")
    require(best is not None, "NEXT maksimum nije pronađen.")

    require(
        len(best) == model.k
        and len(set(best)) == model.k
        and min(best) >= 1
        and max(best) <= model.m,
        "Neispravna NEXT kombinacija.",
    )

    require(
        -math.log(total) - 1e-8 <= best_logp <= 1e-8,
        "Neispravna verovatnoća maksimuma.",
    )

    return best, best_logp


def self_checks():
    say("", "Pokrećem ugrađene matematičke i numeričke provere.")

    model = PyramidModel(m=5, k=2)
    theta = model.initial()

    x = np.array(
        [
            [[0.6, -0.4, 0.6, -0.4, -0.4],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[-0.4, 0.6, -0.4, 0.6, -0.4],
             [0.6, -0.4, 0.6, -0.4, -0.4]],
        ],
        dtype=float,
    )

    y = np.array(
        [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
        dtype=float,
    )

    ridge = 0.03
    _, analytic = objective(theta, model, x, y, ridge)
    numerical = np.empty_like(theta)
    epsilon = 1e-6

    for i in range(len(theta)):
        positive = theta.copy()
        negative = theta.copy()
        positive[i] += epsilon
        negative[i] -= epsilon

        numerical[i] = (
            objective(positive, model, x, y, ridge)[0]
            - objective(negative, model, x, y, ridge)[0]
        ) / (2.0 * epsilon)

    gradient_error = float(np.max(np.abs(analytic - numerical)))

    require(
        gradient_error < 2e-6,
        f"Provera gradijenta nije prošla: {gradient_error:.3e}",
    )

    _, _, decoder, logz, marginals, loggate = distribution_terms(
        model, theta, x
    )

    states = np.array(list(combinations(range(5), 2)), dtype=int)
    occupancy = np.zeros((len(states), 5), dtype=float)

    for i, state in enumerate(states):
        occupancy[i, state] = 1.0

    component = occupancy @ decoder.T - logz

    for bit in range(2):
        probabilities = np.exp(component[:, bit])

        require(
            abs(float(probabilities.sum()) - 1.0) < 1e-12,
            "Normalizacija komponente nije prošla.",
        )

        exact_marginal = probabilities @ occupancy

        require(
            np.max(np.abs(exact_marginal - marginals[bit])) < 1e-12,
            "Provera marginalnih verovatnoća nije prošla.",
        )

    for sample in range(len(x)):
        joint = component + loggate[sample]
        probability = np.exp(logsumexp(joint, axis=1))

        require(
            abs(float(probability.sum()) - 1.0) < 1e-12,
            "Normalizacija mešavine nije prošla.",
        )

    expected_index = int(np.argmax(
        logsumexp(component + loggate[0], axis=1)
    ))

    best, best_logp = exact_next(model, theta, x[:1])
    expected = tuple(int(v) + 1 for v in states[expected_index])

    require(best == expected, "Provera tačnog NEXT maksimuma nije prošla.")

    exact_logp = float(
        logsumexp(component[expected_index] + loggate[0])
    )

    require(
        abs(best_logp - exact_logp) < 1e-12,
        "Provera vrednosti NEXT maksimuma nije prošla.",
    )

    for k in (1, 5):
        log_normalizer, inclusion = subset_statistics(np.zeros(5), k)

        require(
            abs(log_normalizer - math.log(math.comb(5, k))) < 1e-12,
            "Provera granične normalizacije nije prošla.",
        )

        require(
            np.max(np.abs(inclusion - k / 5)) < 1e-12,
            "Provera graničnih margina nije prošla.",
        )

    say(
        "Ugrađene provere su prošle.",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def process(path, rows, occupancy):
    model = PyramidModel(NUMBER_COUNT, DRAW_SIZE)

    targets = np.arange(1, len(occupancy))
    x = contexts(occupancy, targets)
    y = occupancy[targets]

    n = len(y)
    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    folds = (
        (split60, split70),
        (split70, split80),
    )

    say(
        "",
        f"MODEL: {path.name}",
        f"Parametara: {model.size}",
        f"Istorijskih prelaza: {n}",
        "Dva prethodna kola predstavljaju ulaz jednog uzorka.",
        "Završno učenje obuhvata sve raspoložive prelaze.",
    )

    results = []

    for ridge in RIDGES:
        weighted_loss = 0.0
        count = 0

        for fold_number, (train_end, validation_end) in enumerate(folds, 1):
            label = f"Validacija λ={ridge:g}, podela {fold_number}"

            theta = fit(
                model,
                x[:train_end],
                y[:train_end],
                ridge,
                label,
            )

            score = nll(
                model,
                theta,
                x[train_end:validation_end],
                y[train_end:validation_end],
            )

            size = validation_end - train_end
            weighted_loss += score * size
            count += size

            say(f"{label} | NLL {score:.9f}")

        validation_nll = weighted_loss / count
        results.append((validation_nll, ridge))

        say(
            f"λ={ridge:g} | zajednički validacioni NLL"
            f" {validation_nll:.9f}"
        )

    validation_nll, ridge = min(results, key=lambda item: (item[0], item[1]))

    say(
        "",
        f"Izabrano λ: {ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    theta_test = fit(
        model,
        x[:split80],
        y[:split80],
        ridge,
        "Model za odvojeni test",
    )

    test_nll = nll(
        model,
        theta_test,
        x[split80:],
        y[split80:],
    )

    say(
        f"Odvojeni test | prelaza {n - split80}",
        f"Odvojeni test | NLL cele sedmorke {test_nll:.9f}",
    )

    theta = fit(
        model,
        x,
        y,
        ridge,
        "Završno učenje na celoj istoriji",
    )

    next_context = contexts(
        occupancy,
        np.array([len(occupancy)]),
    )

    logits, _, decoder, logz, marginals, loggate = distribution_terms(
        model, theta, next_context
    )

    q = float(expit(logits[0]))
    next_marginal = (1.0 - q) * marginals[0] + q * marginals[1]

    require(
        abs(float(next_marginal.sum()) - DRAW_SIZE) < 1e-8,
        "NEXT margine nemaju očekivani zbir.",
    )

    # KL divergence between the two bit-conditioned set distributions.
    kl01 = float(
        np.dot(marginals[0], decoder[0] - decoder[1])
        - logz[0] + logz[1]
    )

    kl10 = float(
        np.dot(marginals[1], decoder[1] - decoder[0])
        - logz[1] + logz[0]
    )

    require(
        min(kl01, kl10) >= -1e-8,
        "Neispravna KL divergencija dekodera.",
    )

    previous_targets = np.arange(max(1, len(occupancy) - 20), len(occupancy))
    previous_contexts = contexts(occupancy, previous_targets)
    previous_logits, _ = model.encode(theta, previous_contexts)
    previous_q = expit(previous_logits)

    q_range = float(np.ptp(np.concatenate((previous_q, [q]))))

    say(
        "",
        f"Verovatnoća Alice bita 1 za NEXT: {q:.9f}",
        f"Raspon te verovatnoće u poslednjim kontekstima: {q_range:.9f}",
        f"KL dekodera 0 prema 1: {max(0.0, kl01):.9f}",
        f"KL dekodera 1 prema 0: {max(0.0, kl10):.9f}",
        "Pokrećem potpunu pretragu svih legalnih sedmorki.",
    )

    next_numbers, logp = exact_next(model, theta, next_context)

    warnings = []

    if q_range < 1e-4:
        warnings.append(
            "Alice poruka slabo reaguje na poslednje istorijske kontekste."
        )

    if max(kl01, kl10) < 1e-6:
        warnings.append(
            "Dva Bobova dekodera su gotovo ista; binarna poruka malo menja izlaz."
        )

    if longest_run(next_numbers) >= 5:
        warnings.append(
            "NEXT sadrži najmanje pet uzastopnih brojeva."
        )

    if next_numbers[-1] - next_numbers[0] <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva nalazi se u rasponu od deset brojeva."
        )

    repetitions = sum(row == next_numbers for row in rows)

    if repetitions:
        warnings.append(
            f"NEXT se tačno ponavlja u istoriji {repetitions} puta."
        )

    say(
        "",
        f"NEXT {path.name}: "
        + " ".join(f"{number:02d}" for number in next_numbers),
        f"Log-verovatnoća NEXT: {logp:.9f}",
        f"Verovatnoća NEXT prema modelu: {math.exp(logp):.12g}",
    )

    for warning in warnings:
        say(f"DIJAGNOSTIKA: {warning}")

    return {
        "name": path.name,
        "next": next_numbers,
        "test_nll": test_nll,
        "validation_nll": validation_nll,
        "ridge": ridge,
        "warnings": warnings,
    }


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    say(
        "Loto 7/39 — V1",
        "Klasična piramidalna regresija raspodele.",
        "Alice koristi istoriju, Bob binarnu poruku i indeks broja.",
        "Oba moguća bita sabiraju se analitički.",
        "Bez slučajnog generisanja, šuma i mešanja istorije.",
        "Bez PR resursa i kvantnih amplituda.",
        "Tri CSV-a obrađuju se potpuno zasebno.",
        "NEXT je tačan maksimum naučene mešavine raspodela.",
        "Model ima jednu binarnu skrivenu poruku.",
        "Unutar svake komponente zavisnost brojeva potiče iz uslova tačno sedam.",
        "Optimizacija je nekonveksna; globalni optimum parametara nije garantovan.",
    )

    datasets = []
    hashes = []

    for path in FILES:
        require(path.is_file(), f"Fajl nije pronađen: {path}")
        rows, occupancy, digest = load_csv(path)
        datasets.append((path, rows, occupancy))
        hashes.append(digest)

    require(
        len(set(hashes)) == len(hashes),
        "Dva ulaza imaju isti sadržaj. Proveri putanje.",
    )

    self_checks()

    results = [
        process(path, rows, occupancy)
        for path, rows, occupancy in datasets
    ]

    say("", "ZAVRŠNA TRI NEXT REZULTATA")

    for result in results:
        say(
            f"{result['name']}: "
            + " ".join(f"{number:02d}" for number in result["next"]),
            f"Test NLL: {result['test_nll']:.9f}",
            f"Izabrano λ: {result['ridge']:g}",
        )

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if results[i]["next"] == results[j]["next"]:
                say(
                    "DIJAGNOSTIKA: ista NEXT kombinacija za",
                    results[i]["name"],
                    results[j]["name"],
                    "Proveri ispis osetljivosti Alice poruke i razlike dekodera.",
                    "Rezultat nije promenjen radi veštačkog razlikovanja.",
                )

    say(
        "",
        "Obrada je završena.",
        "Test NLL meri raspodelu cele sedmorke na odvojenom delu istorije.",
        "Tačna NEXT pretraga potvrđuje maksimum za naučene parametre.",
        "Ona sama ne potvrđuje prediktivnu prednost modela.",
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
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )
        traceback.print_exc()
        sys.exit(1)



"""
Loto 7/39 — V1
Klasična piramidalna regresija raspodele.
Alice koristi istoriju, Bob binarnu poruku i indeks broja.
Oba moguća bita sabiraju se analitički.
Bez slučajnog generisanja, šuma i mešanja istorije.
Bez PR resursa i kvantnih amplituda.
Tri CSV-a obrađuju se potpuno zasebno.
NEXT je tačan maksimum naučene mešavine raspodela.
Model ima jednu binarnu skrivenu poruku.
Unutar svake komponente zavisnost brojeva potiče iz uslova tačno sedam.
Optimizacija je nekonveksna; globalni optimum parametara nije garantovan.

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

Pokrećem ugrađene matematičke i numeričke provere.
Ugrađene provere su prošle.
Najveća greška gradijenta: 2.640e-10

MODEL: loto7_4710_k86.csv
Parametara: 311
Istorijskih prelaza: 4709
Dva prethodna kola predstavljaju ulaz jednog uzorka.
Završno učenje obuhvata sve raspoložive prelaze.
Validacija λ=0.03, podela 1 | završen | cilj 16.544075276 | gradijent 3.463e-08
Validacija λ=0.03, podela 1 | NLL 16.551164834
Validacija λ=0.03, podela 2 | završen | cilj 16.544710957 | gradijent 1.696e-07
Validacija λ=0.03, podela 2 | NLL 16.548873556
λ=0.03 | zajednički validacioni NLL 16.550019195
Validacija λ=0.3, podela 1 | završen | cilj 16.547379359 | gradijent 1.293e-08
Validacija λ=0.3, podela 1 | NLL 16.548731472
Validacija λ=0.3, podela 2 | završen | cilj 16.547559720 | gradijent 1.119e-08
Validacija λ=0.3, podela 2 | NLL 16.548058812
λ=0.3 | zajednički validacioni NLL 16.548395142
Validacija λ=3, podela 1 | završen | cilj 16.548500649 | gradijent 2.047e-09
Validacija λ=3, podela 1 | NLL 16.548666173
Validacija λ=3, podela 2 | završen | cilj 16.548526691 | gradijent 1.896e-09
Validacija λ=3, podela 2 | NLL 16.548494292
λ=3 | zajednički validacioni NLL 16.548580232

Izabrano λ: 0.3
Validacioni NLL: 16.548395142
Model za odvojeni test | završen | cilj 16.547589055 | gradijent 8.627e-09
Odvojeni test | prelaza 942
Odvojeni test | NLL cele sedmorke 16.547272099
Završno učenje na celoj istoriji | završen | cilj 16.547479252 | gradijent 1.684e-08

Verovatnoća Alice bita 1 za NEXT: 0.511541744
Raspon te verovatnoće u poslednjim kontekstima: 0.047800342
KL dekodera 0 prema 1: 0.009233540
KL dekodera 1 prema 0: 0.009251254
Pokrećem potpunu pretragu svih legalnih sedmorki.
Tačna NEXT pretraga: 1000000 / 15380937
Tačna NEXT pretraga: 2000000 / 15380937
Tačna NEXT pretraga: 3000000 / 15380937
Tačna NEXT pretraga: 4000000 / 15380937
Tačna NEXT pretraga: 5000000 / 15380937
Tačna NEXT pretraga: 6000000 / 15380937
Tačna NEXT pretraga: 7000000 / 15380937
Tačna NEXT pretraga: 8000000 / 15380937
Tačna NEXT pretraga: 9000000 / 15380937
Tačna NEXT pretraga: 10000000 / 15380937
Tačna NEXT pretraga: 11000000 / 15380937
Tačna NEXT pretraga: 12000000 / 15380937
Tačna NEXT pretraga: 13000000 / 15380937
Tačna NEXT pretraga: 14000000 / 15380937
Tačna NEXT pretraga: 15000000 / 15380937

NEXT loto7_4710_k86.csv: 08 10 11 23 25 26 39
Log-verovatnoća NEXT: -16.445788097
Verovatnoća NEXT prema modelu: 7.20584604634e-08

MODEL: loto7_4710_k86_loto_2977.csv
Parametara: 311
Istorijskih prelaza: 2976
Dva prethodna kola predstavljaju ulaz jednog uzorka.
Završno učenje obuhvata sve raspoložive prelaze.
Validacija λ=0.03, podela 1 | završen | cilj 16.540696591 | gradijent 6.276e-08
Validacija λ=0.03, podela 1 | NLL 16.548880421
Validacija λ=0.03, podela 2 | završen | cilj 16.541219381 | gradijent 3.177e-08
Validacija λ=0.03, podela 2 | NLL 16.555998635
λ=0.03 | zajednički validacioni NLL 16.552433546
Validacija λ=0.3, podela 1 | završen | cilj 16.546501530 | gradijent 2.419e-08
Validacija λ=0.3, podela 1 | NLL 16.547394938
Validacija λ=0.3, podela 2 | završen | cilj 16.546615763 | gradijent 1.471e-08
Validacija λ=0.3, podela 2 | NLL 16.549789751
λ=0.3 | zajednički validacioni NLL 16.548590332
Validacija λ=3, podela 1 | završen | cilj 16.548468252 | gradijent 2.402e-09
Validacija λ=3, podela 1 | NLL 16.548355294
Validacija λ=3, podela 2 | završen | cilj 16.548457401 | gradijent 1.825e-09
Validacija λ=3, podela 2 | NLL 16.548902366
λ=3 | zajednički validacioni NLL 16.548628370

Izabrano λ: 0.3
Validacioni NLL: 16.548590332
Model za odvojeni test | završen | cilj 16.546989347 | gradijent 1.057e-08
Odvojeni test | prelaza 596
Odvojeni test | NLL cele sedmorke 16.547784656
Završno učenje na celoj istoriji | završen | cilj 16.547058240 | gradijent 2.699e-08

Verovatnoća Alice bita 1 za NEXT: 0.524304061
Raspon te verovatnoće u poslednjim kontekstima: 0.043257491
KL dekodera 0 prema 1: 0.009144499
KL dekodera 1 prema 0: 0.009161867
Pokrećem potpunu pretragu svih legalnih sedmorki.
Tačna NEXT pretraga: 1000000 / 15380937
Tačna NEXT pretraga: 2000000 / 15380937
Tačna NEXT pretraga: 3000000 / 15380937
Tačna NEXT pretraga: 4000000 / 15380937
Tačna NEXT pretraga: 5000000 / 15380937
Tačna NEXT pretraga: 6000000 / 15380937
Tačna NEXT pretraga: 7000000 / 15380937
Tačna NEXT pretraga: 8000000 / 15380937
Tačna NEXT pretraga: 9000000 / 15380937
Tačna NEXT pretraga: 10000000 / 15380937
Tačna NEXT pretraga: 11000000 / 15380937
Tačna NEXT pretraga: 12000000 / 15380937
Tačna NEXT pretraga: 13000000 / 15380937
Tačna NEXT pretraga: 14000000 / 15380937
Tačna NEXT pretraga: 15000000 / 15380937

NEXT loto7_4710_k86_loto_2977.csv: 08 22 23 24 26 38 39
Log-verovatnoća NEXT: -16.436775652
Verovatnoća NEXT prema modelu: 7.27108186186e-08

MODEL: loto7_4710_k86_loto_plus_1733.csv
Parametara: 311
Istorijskih prelaza: 1732
Dva prethodna kola predstavljaju ulaz jednog uzorka.
Završno učenje obuhvata sve raspoložive prelaze.
Validacija λ=0.03, podela 1 | završen | cilj 16.531957826 | gradijent 8.729e-08
Validacija λ=0.03, podela 1 | NLL 16.556239278
Validacija λ=0.03, podela 2 | završen | cilj 16.534779558 | gradijent 5.152e-08
Validacija λ=0.03, podela 2 | NLL 16.560741877
λ=0.03 | zajednički validacioni NLL 16.558490578
Validacija λ=0.3, podela 1 | završen | cilj 16.544042064 | gradijent 6.591e-08
Validacija λ=0.3, podela 1 | NLL 16.548564558
Validacija λ=0.3, podela 2 | završen | cilj 16.544820163 | gradijent 4.430e-08
Validacija λ=0.3, podela 2 | NLL 16.550135644
λ=0.3 | zajednički validacioni NLL 16.549350101
Validacija λ=3, podela 1 | završen | cilj 16.547999565 | gradijent 7.575e-09
Validacija λ=3, podela 1 | NLL 16.548835451
Validacija λ=3, podela 2 | završen | cilj 16.548148933 | gradijent 9.614e-09
Validacija λ=3, podela 2 | NLL 16.548838447
λ=3 | zajednički validacioni NLL 16.548836949

Izabrano λ: 3
Validacioni NLL: 16.548836949
Model za odvojeni test | završen | cilj 16.548231628 | gradijent 6.973e-09
Odvojeni test | prelaza 347
Odvojeni test | NLL cele sedmorke 16.548396941
Završno učenje na celoj istoriji | završen | cilj 16.548276251 | gradijent 5.387e-09

Verovatnoća Alice bita 1 za NEXT: 0.513220578
Raspon te verovatnoće u poslednjim kontekstima: 0.048536950
KL dekodera 0 prema 1: 0.009276513
KL dekodera 1 prema 0: 0.009295264
Pokrećem potpunu pretragu svih legalnih sedmorki.
Tačna NEXT pretraga: 1000000 / 15380937
Tačna NEXT pretraga: 2000000 / 15380937
Tačna NEXT pretraga: 3000000 / 15380937
Tačna NEXT pretraga: 4000000 / 15380937
Tačna NEXT pretraga: 5000000 / 15380937
Tačna NEXT pretraga: 6000000 / 15380937
Tačna NEXT pretraga: 7000000 / 15380937
Tačna NEXT pretraga: 8000000 / 15380937
Tačna NEXT pretraga: 9000000 / 15380937
Tačna NEXT pretraga: 10000000 / 15380937
Tačna NEXT pretraga: 11000000 / 15380937
Tačna NEXT pretraga: 12000000 / 15380937
Tačna NEXT pretraga: 13000000 / 15380937
Tačna NEXT pretraga: 14000000 / 15380937
Tačna NEXT pretraga: 15000000 / 15380937

NEXT loto7_4710_k86_loto_plus_1733.csv: 10 11 12 25 26 27 39
Log-verovatnoća NEXT: -16.498419751
Verovatnoća NEXT prema modelu: 6.83639805929e-08




ZAVRŠNA TRI NEXT REZULTATA
loto7_4710_k86.csv: 08 10 11 23 25 26 39
Test NLL: 16.547272099
Izabrano λ: 0.3
loto7_4710_k86_loto_2977.csv: 08 22 23 24 26 38 39
Test NLL: 16.547784656
Izabrano λ: 0.3
loto7_4710_k86_loto_plus_1733.csv: 10 11 12 25 26 27 39
Test NLL: 16.548396941
Izabrano λ: 3

Obrada je završena.
Test NLL meri raspodelu cele sedmorke na odvojenom delu istorije.
Tačna NEXT pretraga potvrđuje maksimum za naučene parametre.
Ona sama ne potvrđuje prediktivnu prednost modela.
"""



"""
Obnova poznatog polja. Alice vidi polje, Bob pita za jednu njegovu poziciju. Okruženje generiše slučajna polja; nema učenja prelaza između prethodnog i sledećeg kola.

Neuronski modeli imitiraju učitelja. Alice uči njegove poruke, Bob njegove odluke, uz binarnu unakrsnu entropiju. Regresija raspodele sedmorki.

Piramidalna konstrukcija jeste konkretna ideja: parove informacija sažima kroz više nivoa do jedne poruke, a Bob prati put tražene pozicije.

Za Loto vredi piramidalno sažimanje istorije i dekodiranje po traženom broju. Predviđanje sledećeg kola zahteva novo učenje istorijskih prelaza i zajedničku raspodelu tačno sedam brojeva. 

V1 čuva piramidalno sažimanje i jednu binarnu poruku između Alice i Boba. Alice obrađuje prethodna kola. Bob iz poruke određuje raspodelu naredne sedmorke.

Oba moguća bita računaju se analitički, bez slučajnog uzorkovanja. Ovo je klasična komunikaciona arhitekture
"""
