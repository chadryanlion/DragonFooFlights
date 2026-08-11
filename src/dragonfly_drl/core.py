"""
Tide-Pool Noisy DQN / Fractal-Repertoire / CPG Evaluation Prototype

Dependencies:
    numpy
    scipy
    torch
    scikit-learn

The neuroscience overlay is intentionally represented as a computational
control architecture, not as a claim that these modules are literal
one-to-one mappings onto mammalian brain regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------
# 1. TIDE-POOL RADIAL-BASIS STATE REPRESENTATION
# ---------------------------------------------------------------------

@dataclass
class TidePool:
    center: np.ndarray
    radius: float
    height: float
    non_drop: float
    flake_koch: float
    purplex: float


class TidePoolRBF:
    """
    Maps a continuous environmental state into a radial basis representation.

    The environmental variables are intentionally explicit:
        pool height
        pool radius
        non-drop
        flake/Koch complexity
        purplex complexity
    """

    def __init__(
        self,
        centers: np.ndarray,
        sigma: float = 1.0,
        pools: List[TidePool] | None = None,
    ):
        self.centers = np.asarray(centers, dtype=np.float32)
        self.sigma = float(sigma)
        self.pools = pools or []

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        d2 = ((self.centers - x[None, :]) ** 2).sum(axis=1)
        rbf = np.exp(-d2 / (2.0 * self.sigma ** 2))

        return rbf / (rbf.sum() + 1e-8)


# ---------------------------------------------------------------------
# 2. STATE COUNT CONSTRUCTOR
# ---------------------------------------------------------------------

class StateCountConstructor:
    """
    Builds empirical state visitation counts.

    A count-based bonus can replace epsilon-heavy exploration:

        bonus(s) = beta / sqrt(N(s) + 1)

    The representation can be a quantized RBF vector or a discrete state ID.
    """

    def __init__(self, beta: float = 0.05, decimals: int = 2):
        self.beta = beta
        self.decimals = decimals
        self.counts = defaultdict(int)

    def key(self, state: np.ndarray) -> Tuple:
        return tuple(np.round(state, self.decimals).tolist())

    def update(self, state: np.ndarray) -> int:
        k = self.key(state)
        self.counts[k] += 1
        return self.counts[k]

    def bonus(self, state: np.ndarray) -> float:
        k = self.key(state)
        return self.beta / np.sqrt(self.counts[k] + 1.0)


# ---------------------------------------------------------------------
# 3. NOISY LINEAR / NOISY DQN
# ---------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma0: float = 0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mu_w = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.sigma_w = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        self.register_buffer(
            "eps_i", torch.zeros(in_features)
        )
        self.register_buffer(
            "eps_j", torch.zeros(out_features)
        )

        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / np.sqrt(self.in_features)

        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.constant_(self.sigma_w,
                          self.sigma0 / np.sqrt(self.in_features))

        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_b,
                          self.sigma0 / np.sqrt(self.out_features))

    @staticmethod
    def _f(x):
        return x.sign() * torch.sqrt(x.abs())

    def reset_noise(self):
        eps_i = torch.randn(self.in_features, device=self.mu_w.device)
        eps_j = torch.randn(self.out_features, device=self.mu_w.device)

        self.eps_i.copy_(self._f(eps_i))
        self.eps_j.copy_(self._f(eps_j))

    def forward(self, x):
        eps_w = self.eps_j[:, None] * self.eps_i[None, :]
        eps_b = self.eps_j

        w = self.mu_w + self.sigma_w * eps_w
        b = self.mu_b + self.sigma_b * eps_b

        return F.linear(x, w, b)


class TidePoolNoisyDQN(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        sigma0: float = 0.5,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, 1, sigma0),
        )

        self.advantage = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, action_dim, sigma0),
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        h = self.fc(x)

        value = self.value(h)
        advantage = self.advantage(h)

        return value + advantage - advantage.mean(
            dim=-1,
            keepdim=True
        )

    @torch.no_grad()
    def greedy_action(self, state):
        q = self.forward(state)
        return q.argmax(dim=-1)


# ---------------------------------------------------------------------
# 4. RHEOSTAT NOISE CALIBRATION
# ---------------------------------------------------------------------

def rheostat_objective(
    log_lambda: np.ndarray,
    q_mean: np.ndarray,
    q_std: np.ndarray,
    target_entropy: float,
    kl_uniform_penalty: float = 0.25,
):
    """
    Scalar calibration objective.

    lambda controls the amount of NoisyNet perturbation.

    The optimization balances:
        - action entropy
        - excessive approach to the uniform-policy regime
        - desired exploration entropy

    scipy's nonlinear conjugate-gradient solver is used below.
    """

    lam = np.exp(float(log_lambda[0]))

    logits = q_mean / (1.0 + lam * q_std + 1e-8)
    p = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)

    entropy = -(p * torch.log(p + 1e-8)).sum().item()

    uniform = torch.ones_like(p) / len(p)

    kl_to_uniform = (
        p * torch.log((p + 1e-8) / uniform)
    ).sum().item()

    return (
        (entropy - target_entropy) ** 2
        + kl_uniform_penalty * kl_to_uniform
    )


def calibrate_rheostat(
    q_mean: np.ndarray,
    q_std: np.ndarray,
    target_entropy: float,
):
    result = minimize(
        rheostat_objective,
        x0=np.array([0.0]),
        args=(q_mean, q_std, target_entropy),
        method="CG",
    )

    return float(np.exp(result.x[0])), result


# ---------------------------------------------------------------------
# 5. COASTLINE / FRACTAL UPDATE DISCRIMINATOR
# ---------------------------------------------------------------------

def coastline_features(
    pool_height: float,
    radius: float,
    non_drop: float,
    flake_koch: float,
    purplex: float,
) -> np.ndarray:

    circumference = 2.0 * np.pi * radius

    aspect = pool_height / (radius + 1e-6)

    return np.array([
        pool_height,
        radius,
        non_drop,
        flake_koch,
        purplex,
        circumference,
        aspect,
        flake_koch * radius,
        purplex * non_drop,
        np.log1p(abs(flake_koch)),
        np.log1p(abs(purplex)),
    ], dtype=np.float32)


class CoastlineDiscriminator(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def coastline_update(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """
    Fractal-update feature:

        delta = after - before

    This lets the discriminator distinguish genuine geometric
    environmental change from static morphology.
    """

    return after - before


# ---------------------------------------------------------------------
# 6. FLIGHT REPERTOIRE REPRESENTATION
# ---------------------------------------------------------------------

@dataclass
class FlightState:
    speed: float
    heading: float
    altitude: float
    turn_rate: float
    acceleration: float
    neighbor_distance: float
    obstacle_distance: float
    energy: float


def flight_vector(x: FlightState) -> np.ndarray:
    return np.array([
        x.speed,
        np.sin(x.heading),
        np.cos(x.heading),
        x.altitude,
        x.turn_rate,
        x.acceleration,
        x.neighbor_distance,
        x.obstacle_distance,
        x.energy,
    ], dtype=np.float32)


# ---------------------------------------------------------------------
# 7. HIERARCHICAL KL MAP
# ---------------------------------------------------------------------

def normalized_histogram(
    vectors: np.ndarray,
    bins: int = 16,
    low: float = -5.0,
    high: float = 5.0,
) -> np.ndarray:

    vectors = np.asarray(vectors)

    hist = []
    for i in range(vectors.shape[1]):
        h, _ = np.histogram(
            vectors[:, i],
            bins=bins,
            range=(low, high),
            density=False,
        )
        hist.append(h)

    h = np.concatenate(hist).astype(np.float64)
    h += 1e-8

    return h / h.sum()


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12

    p /= p.sum()
    q /= q.sum()

    return np.sum(p * np.log(p / q))


def hierarchical_kl_map(
    repertoires: Dict[str, np.ndarray],
):
    names = list(repertoires)

    distributions = {
        k: normalized_histogram(v)
        for k, v in repertoires.items()
    }

    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Jensen-Shannon-like symmetric KL
            matrix[i, j] = 0.5 * (
                kl_divergence(
                    distributions[names[i]],
                    distributions[names[j]],
                )
                +
                kl_divergence(
                    distributions[names[j]],
                    distributions[names[i]],
                )
            )

    condensed = squareform(matrix, checks=False)
    hierarchy = linkage(condensed, method="average")

    order = leaves_list(hierarchy)

    return matrix, hierarchy, [names[i] for i in order]


# ---------------------------------------------------------------------
# 8. FIVE-COMPONENT CPG CONTROL ARCHITECTURE
# ---------------------------------------------------------------------

@dataclass
class CPGOutput:
    amplitude: float
    frequency: float
    phase: float
    gain: float


class CPGModule:

    def __init__(self, name: str):
        self.name = name

    def calibrate(
        self,
        error: float,
        context: float,
    ) -> CPGOutput:

        # Minimal adaptive oscillator controller.
        gain = 1.0 / (1.0 + abs(error))
        frequency = 1.0 + 0.5 * np.tanh(context)
        amplitude = gain * (1.0 + 0.25 * np.tanh(context))
        phase = np.arctan2(context, error + 1e-6)

        return CPGOutput(
            amplitude=float(amplitude),
            frequency=float(frequency),
            phase=float(phase),
            gain=float(gain),
        )


class FiveCPGController:

    """
    Five computational control modules.

    1. wingbeat / propulsion
    2. heading / steering
    3. altitude / vertical stabilization
    4. obstacle / escape
    5. swarm coordination / landing

    These are functional modules, not literal anatomical nuclei.
    """

    def __init__(self):
        self.modules = {
            "propulsion": CPGModule("propulsion"),
            "steering": CPGModule("steering"),
            "altitude": CPGModule("altitude"),
            "escape": CPGModule("escape"),
            "swarm": CPGModule("swarm"),
        }

    def calibrate(self, errors, contexts):

        output = {}

        for name, module in self.modules.items():
            output[name] = module.calibrate(
                error=float(errors[name]),
                context=float(contexts[name]),
            )

        return output


# ---------------------------------------------------------------------
# 9. COMPUTATIONAL "BRAIN OVERLAY"
# ---------------------------------------------------------------------

class MammalianControlOverlay:
    """
    Functional computational overlay.

    Basal ganglia:
        action gating

    Cerebellar:
        predictive error correction

    ACC:
        conflict / monitoring cost

    dlPFC:
        planning / working-state persistence

    vmPFC:
        value/context integration

    Amygdala:
        salience / rapid defensive weighting
    """

    def __init__(self):
        self.gain = {
            "basal_ganglia": 1.0,
            "cerebellar": 1.0,
            "ACC": 1.0,
            "dlPFC": 1.0,
            "vmPFC": 1.0,
            "amygdala": 1.0,
        }

    def apply(
        self,
        action_values: np.ndarray,
        prediction_error: float,
        conflict: float,
        salience: float,
        context_value: float,
    ):

        gated = action_values * self.gain["basal_ganglia"]

        # Predictive error correction.
        gated -= (
            self.gain["cerebellar"]
            * prediction_error
        )

        # Conflict penalty.
        gated -= (
            self.gain["ACC"]
            * conflict
        )

        # Planning/value contribution.
        gated += (
            self.gain["dlPFC"]
            * context_value
        )

        gated += (
            self.gain["vmPFC"]
            * context_value
        )

        # Rapid salience term.
        gated += (
            self.gain["amygdala"]
            * salience
        )

        return gated


# ---------------------------------------------------------------------
# 10. MATURATION / CALIBRATION SCHEDULE
# ---------------------------------------------------------------------

def maturation_gain(
    developmental_fraction: float,
    early_gain: float,
    adult_gain: float,
):

    x = np.clip(developmental_fraction, 0.0, 1.0)

    # Smooth developmental interpolation.
    s = x * x * (3.0 - 2.0 * x)

    return early_gain + s * (adult_gain - early_gain)


def apply_maturation(
    overlay: MammalianControlOverlay,
    developmental_fraction: float,
):

    for name in overlay.gain:
        overlay.gain[name] = maturation_gain(
            developmental_fraction,
            early_gain=0.5,
            adult_gain=1.0,
        )

    return overlay


# ---------------------------------------------------------------------
# 11. NO-FLIGHT BASELINE
# ---------------------------------------------------------------------

def no_flight_baseline(n: int = 1000) -> np.ndarray:

    """
    Baseline repertoire representing absence of organized flight.

    It intentionally has:
        near-zero speed
        low heading change
        no altitude trajectory
        minimal coordinated acceleration
    """

    rng = np.random.default_rng(7)

    x = np.zeros((n, 9), dtype=np.float32)

    x[:, 0] = rng.normal(0.0, 0.02, n)
    x[:, 1] = rng.normal(0.0, 0.02, n)
    x[:, 2] = 1.0
    x[:, 3] = rng.normal(0.0, 0.02, n)
    x[:, 4] = rng.normal(0.0, 0.02, n)
    x[:, 5] = rng.normal(0.0, 0.02, n)
    x[:, 6] = 1.0
    x[:, 7] = 1.0
    x[:, 8] = 1.0

    return x


# ---------------------------------------------------------------------
# 12. COMPLETE EVALUATION
# ---------------------------------------------------------------------

def evaluate_flight_system(
    dragonfly: np.ndarray,
    monarch: np.ndarray,
):

    no_flight = no_flight_baseline(
        min(len(dragonfly), len(monarch))
    )

    repertoires = {
        "dragonfly_swarm": dragonfly,
        "monarch_migration": monarch,
        "no_flight": no_flight,
    }

    kl_matrix, hierarchy, order = hierarchical_kl_map(
        repertoires
    )

    return {
        "KL_matrix": kl_matrix,
        "hierarchical_order": order,
        "hierarchy": hierarchy,
    }
