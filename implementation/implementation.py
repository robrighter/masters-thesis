"""
Recursive Computation of Compound Random Variables
with Finite Mixtures of Panjer Class Distributions

This module implements the recursive method from:
"Recursive Methods for Mixed Poisson Distributions and Compound Random Variables"

The Panjer class includes: Poisson, Binomial, and Negative Binomial distributions.
"""

import math
from typing import List, Tuple, Dict
from enum import Enum

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DistributionType(Enum):
    POISSON = "poisson"
    BINOMIAL = "binomial"
    NEGATIVE_BINOMIAL = "negative_binomial"


class MixedCompoundRV:
    """
    Computes P(S_N = k) for a compound random variable S_N = sum(X_i, i=1..N)
    where N follows a finite mixture of Panjer class distributions.
    
    Parameters
    ----------
    components : List of tuples (dist_type, params, weight)
        - dist_type: DistributionType enum
        - params: dict with distribution parameters
            - Poisson: {"lambda": λ}
            - Binomial: {"r": r, "p": p}
            - Negative Binomial: {"r": r, "p": p}
        - weight: mixing weight β_i (must sum to 1)
    
    claim_probs : dict mapping positive integers to probabilities
        P(X_1 = i) = α_i for i = 1, 2, 3, ...
    """
    
    def __init__(self, components: List[Tuple], claim_probs: dict):
        self.components = components
        self.claim_probs = claim_probs
        self._validate_inputs()
        self._cache = {}
        self._weights_cache = {}
        self._op_count = 0
        self._op_count_uncached = 0
    
    def _validate_inputs(self):
        """Validate that weights sum to 1 and claim probs sum to 1."""
        weight_sum = sum(c[2] for c in self.components)
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-9):
            raise ValueError(f"Mixing weights must sum to 1, got {weight_sum}")
        
        prob_sum = sum(self.claim_probs.values())
        if not math.isclose(prob_sum, 1.0, rel_tol=1e-9):
            raise ValueError(f"Claim probabilities must sum to 1, got {prob_sum}")
        
        if any(i <= 0 for i in self.claim_probs.keys()):
            raise ValueError("Claim sizes must be positive integers")
    
    def _get_expected_value_component(self, dist_type: DistributionType, 
                                       params: dict, s: int) -> float:
        """
        Compute E[N_j^(s)] for a single component at recursion level s.
        
        For Poisson: E[N] = λ (invariant with s)
        For Binomial: E[N^(s)] = (r - s) * p
        For Negative Binomial: E[N^(s)] = (r + s) * (1 - p) / p
        """
        if dist_type == DistributionType.POISSON:
            return params["lambda"]
        elif dist_type == DistributionType.BINOMIAL:
            r, p = params["r"], params["p"]
            return (r - s) * p
        elif dist_type == DistributionType.NEGATIVE_BINOMIAL:
            r, p = params["r"], params["p"]
            return (r + s) * (1 - p) / p
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def _get_weights_at_level(self, s: int) -> List[float]:
        """
        Compute the mixing weights β_j^(s) at recursion level s.

        β_j^(s+1) = β_j^(s) * E[N_j^(s)] / E[N^(s)]
        """
        if s in self._weights_cache:
            return self._weights_cache[s]

        if s == 0:
            result = [c[2] for c in self.components]
            self._weights_cache[0] = result
            return result
        
        # Get weights at level s-1
        prev_weights = self._get_weights_at_level(s - 1)
        
        # Compute E[N_j^(s-1)] for each component
        component_expectations = []
        for i, (dist_type, params, _) in enumerate(self.components):
            e_nj = self._get_expected_value_component(dist_type, params, s - 1)
            # Clamp to zero for binomial when r - s < 0
            e_nj = max(0.0, e_nj)
            component_expectations.append(e_nj)
        
        # Compute E[N^(s-1)] = sum(β_j^(s-1) * E[N_j^(s-1)])
        total_expectation = sum(
            prev_weights[j] * component_expectations[j]
            for j in range(len(self.components))
        )
        
        # Handle edge case where total expectation is zero
        if total_expectation == 0:
            return prev_weights
        
        # Compute new weights
        new_weights = [
            prev_weights[j] * component_expectations[j] / total_expectation
            for j in range(len(self.components))
        ]

        self._weights_cache[s] = new_weights
        return new_weights
    
    def _get_expected_value_mixture(self, s: int) -> float:
        """
        Compute E[N^(s)] for the mixture at recursion level s.
        
        E[N^(s)] = sum(β_j^(s) * E[N_j^(s)])
        """
        weights = self._get_weights_at_level(s)
        total = 0.0
        for j, (dist_type, params, _) in enumerate(self.components):
            e_nj = self._get_expected_value_component(dist_type, params, s)
            # Clamp to zero for binomial when r - s < 0
            e_nj = max(0.0, e_nj)
            total += weights[j] * e_nj
        return total
    
    def _pmf_component(self, dist_type: DistributionType, params: dict,
                       s: int, n: int) -> float:
        """
        Compute P(N_j^(s) = n) for a single component.
        
        At recursion level s:
        - Poisson: unchanged, P(N=n) = e^(-λ) * λ^n / n!
        - Binomial: r becomes (r - s)
        - Negative Binomial: r becomes (r + s)
        """
        if n < 0:
            return 0.0
        
        if dist_type == DistributionType.POISSON:
            lam = params["lambda"]
            return math.exp(-lam) * (lam ** n) / math.factorial(n)
        
        elif dist_type == DistributionType.BINOMIAL:
            r_eff = params["r"] - s
            p = params["p"]
            if n > r_eff or r_eff < 0:
                return 0.0
            return (math.comb(r_eff, n) * (p ** n) * 
                    ((1 - p) ** (r_eff - n)))
        
        elif dist_type == DistributionType.NEGATIVE_BINOMIAL:
            r_eff = params["r"] + s
            p = params["p"]
            return (math.comb(n + r_eff - 1, n) * (p ** r_eff) * 
                    ((1 - p) ** n))
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def _pmf_mixture_at_level(self, s: int, n: int) -> float:
        """
        Compute P(N^(s) = n) for the mixture at recursion level s.
        
        P(N^(s) = n) = sum(β_j^(s) * P_rj(N = n))
        """
        weights = self._get_weights_at_level(s)
        total = 0.0
        for j, (dist_type, params, _) in enumerate(self.components):
            total += weights[j] * self._pmf_component(dist_type, params, s, n)
        return total
    
    def compute_prob(self, k: int, s: int = 0) -> float:
        """
        Compute P_s(S_N = k) using the recursive formula from Corollary 3.

        P_s(S_N = k) = (E[N^(s)] / k) * sum(i * α_i * P_{s+1}(S_N = k - i), i=1..k)

        Base case: P_s(S_N = 0) = P(N^(s) = 0)

        Parameters
        ----------
        k : int
            The target value for S_N
        s : int
            The recursion level (default 0 for initial call)

        Returns
        -------
        float
            P(S_N = k) when s=0
        """
        # Check cache
        cache_key = (k, s)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Base case
        if k == 0:
            result = self._pmf_mixture_at_level(s, 0)
            self._cache[cache_key] = result
            return result

        if k < 0:
            return 0.0

        # Recursive case
        e_n = self._get_expected_value_mixture(s)

        total = 0.0
        max_claim = max(self.claim_probs.keys())

        for i in range(1, min(k, max_claim) + 1):
            if i in self.claim_probs:
                alpha_i = self.claim_probs[i]
                p_next = self.compute_prob(k - i, s + 1)
                total += i * alpha_i * p_next
                self._op_count += 1  # count each term in the convolution sum

        result = (e_n / k) * total
        self._cache[cache_key] = result
        return result
    
    def compute_distribution(self, max_k: int) -> dict:
        """
        Compute P(S_N = k) for k = 0, 1, ..., max_k.

        Parameters
        ----------
        max_k : int
            Maximum value of k to compute

        Returns
        -------
        dict
            Mapping from k to P(S_N = k)
        """
        self._cache = {}  # Clear cache for fresh computation
        self._weights_cache = {}
        self._op_count = 0
        return {k: self.compute_prob(k) for k in range(max_k + 1)}
    
    def clear_cache(self):
        """Clear the computation cache and operation counter."""
        self._cache = {}
        self._weights_cache = {}
        self._op_count = 0

    def compute_prob_uncached(self, k: int, s: int = 0) -> float:
        """
        Identical recursive formula to compute_prob but with NO memoization.

        Every subproblem is recomputed from scratch on every call, exposing the
        full exponential call tree.  Only used for counting operations so that
        the benefit of caching can be quantified.
        """
        if k == 0:
            self._op_count_uncached += 1  # base-case evaluation
            return self._pmf_mixture_at_level(s, 0)

        if k < 0:
            return 0.0

        e_n = self._get_expected_value_mixture(s)
        total = 0.0
        max_claim = max(self.claim_probs.keys())

        for i in range(1, min(k, max_claim) + 1):
            if i in self.claim_probs:
                alpha_i = self.claim_probs[i]
                p_next = self.compute_prob_uncached(k - i, s + 1)
                total += i * alpha_i * p_next
                self._op_count_uncached += 1

        return (e_n / k) * total

    def count_ops_uncached(self, k: int) -> int:
        """Return the operation count for compute_prob_uncached(k)."""
        self._op_count_uncached = 0
        self.compute_prob_uncached(k)
        return self._op_count_uncached

    def count_ops_for_k(self, k: int) -> int:
        """
        Return the number of convolution-sum operations needed to compute P(S_N = k)
        from scratch (empty cache).
        """
        self.clear_cache()
        self.compute_prob(k)
        return self._op_count

    def peak_memory_for_k(self, k: int) -> int:
        """
        Return the number of cached values stored after computing P(S_N = k).

        For the recursive method all intermediate results accumulate in the
        cache and are never discarded, so peak memory equals the total number
        of unique (value, level) pairs that were computed.
        """
        self.clear_cache()
        self.compute_prob(k)
        return len(self._cache)


# ============================================================================
# SIMULATION CLASS
# ============================================================================

import random


class CompoundRVSimulator:
    """
    Estimates P(S_N = k) for a compound random variable through Monte Carlo simulation.
    
    Parameters
    ----------
    components : List of tuples (dist_type, params, weight)
        Same format as MixedCompoundRV
    
    claim_probs : dict mapping positive integers to probabilities
        P(X_1 = i) = α_i for i = 1, 2, 3, ...
    
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, components: List[Tuple], claim_probs: dict, seed: int = None):
        self.components = components
        self.claim_probs = claim_probs
        self._claim_values = list(claim_probs.keys())
        self._claim_weights = list(claim_probs.values())
        
        if seed is not None:
            random.seed(seed)
    
    def _sample_component_index(self) -> int:
        """Sample which mixture component to use based on mixing weights."""
        weights = [c[2] for c in self.components]
        return random.choices(range(len(self.components)), weights=weights)[0]
    
    def _sample_n(self, dist_type: DistributionType, params: dict) -> int:
        """
        Sample a value of N from the specified distribution.
        
        Uses inverse transform or direct sampling methods.
        """
        if dist_type == DistributionType.POISSON:
            # Knuth algorithm for Poisson sampling
            lam = params["lambda"]
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            return k - 1
        
        elif dist_type == DistributionType.BINOMIAL:
            r, p = params["r"], params["p"]
            # Sum of r Bernoulli trials
            return sum(1 for _ in range(r) if random.random() < p)
        
        elif dist_type == DistributionType.NEGATIVE_BINOMIAL:
            r, p = params["r"], params["p"]
            # Count failures before r successes
            successes = 0
            failures = 0
            while successes < r:
                if random.random() < p:
                    successes += 1
                else:
                    failures += 1
            return failures
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def _sample_claim(self) -> int:
        """Sample a single claim size X_i."""
        return random.choices(self._claim_values, weights=self._claim_weights)[0]
    
    def sample_s_n(self) -> int:
        """
        Sample a single value of S_N.
        
        1. Select mixture component
        2. Sample N from that component's distribution
        3. Sample N claim sizes and sum them
        """
        # Select component
        comp_idx = self._sample_component_index()
        dist_type, params, _ = self.components[comp_idx]
        
        # Sample N
        n = self._sample_n(dist_type, params)
        
        # Sample and sum claims
        if n == 0:
            return 0
        return sum(self._sample_claim() for _ in range(n))
    
    def estimate_distribution(self, max_k: int, n_simulations: int = 1000000) -> dict:
        """
        Estimate P(S_N = k) for k = 0, 1, ..., max_k via simulation.
        
        Parameters
        ----------
        max_k : int
            Maximum value of k to track
        n_simulations : int
            Number of Monte Carlo samples (default 1,000,000)
        
        Returns
        -------
        dict
            Mapping from k to estimated P(S_N = k)
        """
        counts = {k: 0 for k in range(max_k + 1)}
        
        for _ in range(n_simulations):
            s_n = self.sample_s_n()
            if s_n <= max_k:
                counts[s_n] += 1
        
        return {k: count / n_simulations for k, count in counts.items()}
    
    def estimate_prob(self, k: int, n_simulations: int = 1000000) -> float:
        """
        Estimate P(S_N = k) via simulation.
        
        Parameters
        ----------
        k : int
            Target value
        n_simulations : int
            Number of Monte Carlo samples
        
        Returns
        -------
        float
            Estimated probability
        """
        count = 0
        for _ in range(n_simulations):
            if self.sample_s_n() == k:
                count += 1
        return count / n_simulations


# ============================================================================
# TESTS
# ============================================================================

def test_example_1_poisson():
    """
    Test from thesis Example 1: N ~ Poisson(λ=3)
    Expected: P(S_N = 4) = 1.528521094 * e^(-3)
    """
    print("=" * 60)
    print("Test 1: Poisson Distribution (Example 1 from thesis)")
    print("=" * 60)
    
    # Claim distribution from thesis
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    # Single Poisson with λ=3 (as a mixture with one component)
    components = [
        (DistributionType.POISSON, {"lambda": 3}, 1.0)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 1.528521094 * math.exp(-3)
    
    print(f"Computed P(S_N = 4) = {result:.10f}")
    print(f"Expected P(S_N = 4) = {expected:.10f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-6)


def test_example_2_negative_binomial():
    """
    Test from thesis Example 2: N ~ NegBin(r=6, p=0.6)
    Expected: P(S_N = 4) = 0.05514609309696
    """
    print("=" * 60)
    print("Test 2: Negative Binomial Distribution (Example 2 from thesis)")
    print("=" * 60)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    components = [
        (DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.6}, 1.0)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 0.05514609309696
    
    print(f"Computed P(S_N = 4) = {result:.14f}")
    print(f"Expected P(S_N = 4) = {expected:.14f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-6)


def test_example_3_binomial():
    """
    Test from thesis Example 3: N ~ Bin(r=6, p=0.6)
    Expected: P(S_N = 4) = 0.033548184
    """
    print("=" * 60)
    print("Test 3: Binomial Distribution (Example 3 from thesis)")
    print("=" * 60)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    components = [
        (DistributionType.BINOMIAL, {"r": 6, "p": 0.6}, 1.0)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 0.033548184
    
    print(f"Computed P(S_N = 4) = {result:.12f}")
    print(f"Expected P(S_N = 4) = {expected:.12f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-5)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-5)


def test_example_4_mixed_poisson():
    """
    Test from thesis Example 4/6: Mixed Poisson (λ1=3, λ2=4, β1=0.6, β2=0.4)
    Expected: P(S_N = 4) = 0.9171126562499999*e^(-3) + 0.9568266666666668*e^(-4)
    """
    print("=" * 60)
    print("Test 4: Mixed Poisson Distribution (Example 4/6 from thesis)")
    print("=" * 60)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    components = [
        (DistributionType.POISSON, {"lambda": 3}, 0.6),
        (DistributionType.POISSON, {"lambda": 4}, 0.4)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 0.9171126562499999 * math.exp(-3) + 0.9568266666666668 * math.exp(-4)
    
    print(f"Computed P(S_N = 4) = {result:.12f}")
    print(f"Expected P(S_N = 4) = {expected:.12f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-6)


def test_example_7_mixed_binomial():
    """
    Test from thesis Example 7: Mixed Binomial
    Component 1: Bin(r1=8, p1=0.5), β1=0.6
    Component 2: Bin(r2=10, p2=0.7), β2=0.4
    Expected: P(S_N = 4) = 0.016361573047297497
    """
    print("=" * 60)
    print("Test 5: Mixed Binomial Distribution (Example 7 from thesis)")
    print("=" * 60)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    components = [
        (DistributionType.BINOMIAL, {"r": 8, "p": 0.5}, 0.6),
        (DistributionType.BINOMIAL, {"r": 10, "p": 0.7}, 0.4)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 0.016361573047297497
    
    print(f"Computed P(S_N = 4) = {result:.18f}")
    print(f"Expected P(S_N = 4) = {expected:.18f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-6)


def test_example_8_mixed_negative_binomial():
    """
    Test from thesis Example 8: Mixed Negative Binomial
    Component 1: NegBin(r1=4, p1=0.6), β1=0.5
    Component 2: NegBin(r2=6, p2=0.5), β2=0.5
    Expected: P(S_N = 4) = 0.05679127315 (corrected thesis value)

    The thesis originally displayed 0.05679524977, which was the result of
    propagating rounded intermediate values through the final step.  The
    corrected value 0.05679127315 is confirmed by both the recursive and
    LTP methods and is now reflected in the thesis.
    """
    print("=" * 60)
    print("Test 6: Mixed Negative Binomial (Example 8 from thesis)")
    print("=" * 60)

    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}

    components = [
        (DistributionType.NEGATIVE_BINOMIAL, {"r": 4, "p": 0.6}, 0.5),
        (DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.5}, 0.5)
    ]

    model = MixedCompoundRV(components, claim_probs)

    result = model.compute_prob(4)
    expected = 0.056791273151484377  # exact computed value; thesis displays rounded 0.05679127315

    print(f"Computed P(S_N = 4) = {result:.18f}")
    print(f"Expected P(S_N = 4) = {expected:.18f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")

    return math.isclose(result, expected, rel_tol=1e-6)


def test_example_9_mixed_all_three():
    """
    Test from thesis Example 9: Mixed Poisson, Binomial, and Negative Binomial
    Component 1: Poisson(λ=3), β1=0.4
    Component 2: Binomial(r=6, p=0.5), β2=0.3
    Component 3: NegBin(r=2, p=0.4), β3=0.3
    Expected: P(S_N = 4) = 0.0730437085316264
    """
    print("=" * 60)
    print("Test 7: Mixed All Three Types (Example 9 from thesis)")
    print("=" * 60)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    
    components = [
        (DistributionType.POISSON, {"lambda": 3}, 0.4),
        (DistributionType.BINOMIAL, {"r": 6, "p": 0.5}, 0.3),
        (DistributionType.NEGATIVE_BINOMIAL, {"r": 2, "p": 0.4}, 0.3)
    ]
    
    model = MixedCompoundRV(components, claim_probs)
    
    result = model.compute_prob(4)
    expected = 0.0730437085316264
    
    print(f"Computed P(S_N = 4) = {result:.18f}")
    print(f"Expected P(S_N = 4) = {expected:.18f}")
    print(f"Difference: {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-5)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-5)


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Example 1: Poisson", test_example_1_poisson),
        ("Example 2: Negative Binomial", test_example_2_negative_binomial),
        ("Example 3: Binomial", test_example_3_binomial),
        ("Example 4/6: Mixed Poisson", test_example_4_mixed_poisson),
        ("Example 7: Mixed Binomial", test_example_7_mixed_binomial),
        ("Example 8: Mixed Neg. Binomial", test_example_8_mixed_negative_binomial),
        ("Example 9: Mixed All Three", test_example_9_mixed_all_three),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print("=" * 60)
    
    return passed_count == total_count


# ============================================================================
# SIMULATION VS RECURSIVE COMPARISON
# ============================================================================

def compare_simulation_vs_recursive(name: str, components: List[Tuple],
                                     claim_probs: dict, max_k: int = 10,
                                     n_simulations: int = 500000):
    """
    Compare simulation estimates with both the recursive and LTP exact methods.

    Parameters
    ----------
    name : str
        Description of the test case
    components : list
        Mixture components
    claim_probs : dict
        Claim size distribution
    max_k : int
        Maximum k value to compare
    n_simulations : int
        Number of simulation iterations
    """
    print(f"\n{'=' * 85}")
    print(f"COMPARISON: {name}")
    print(f"Simulations: {n_simulations:,}")
    print(f"{'=' * 85}")

    # Recursive computation
    recursive_model = MixedCompoundRV(components, claim_probs)
    recursive_probs = recursive_model.compute_distribution(max_k)

    # LTP computation
    ltp_model = LTPCompoundRV(components, claim_probs)
    ltp_probs = ltp_model.compute_distribution(max_k)

    # Simulation
    simulator = CompoundRVSimulator(components, claim_probs, seed=42)
    simulated_probs = simulator.estimate_distribution(max_k, n_simulations)

    # Compare results
    print(f"\n{'k':>4}  {'Recursive':>14}  {'LTP':>14}  {'Simulated':>14}"
          f"  {'|Rec-Sim|':>11}  {'|LTP-Sim|':>11}")
    print("-" * 80)

    # Relative error is only meaningful when the exact probability is large enough
    # for the simulation to produce a reliable estimate.  With n_simulations draws,
    # the relative standard error of a proportion p is 1/sqrt(n*p).  We require at
    # least MIN_EXPECTED_COUNT expected events so that 1/sqrt(n*p) <= ~5%, giving a
    # comfortable margin below the 10% pass threshold.
    MIN_EXPECTED_COUNT = 400  # ensures rel std err <= 1/sqrt(400) = 5%
    rel_threshold = MIN_EXPECTED_COUNT / n_simulations

    total_abs_error_rec = 0.0
    total_abs_error_ltp = 0.0
    max_rel_error_rec = 0.0
    max_rel_error_ltp = 0.0
    skipped_k = []

    for k in range(max_k + 1):
        rec = recursive_probs[k]
        ltp = ltp_probs[k]
        sim = simulated_probs[k]
        abs_diff_rec = abs(rec - sim)
        abs_diff_ltp = abs(ltp - sim)

        total_abs_error_rec += abs_diff_rec
        total_abs_error_ltp += abs_diff_ltp

        # Only track relative error when the exact probability is above the
        # reliability threshold; below it the simulation variance dominates.
        if rec >= rel_threshold:
            rel_diff_rec = abs_diff_rec / rec
            rel_diff_ltp = abs_diff_ltp / ltp if ltp > 1e-10 else 0.0
            max_rel_error_rec = max(max_rel_error_rec, rel_diff_rec)
            max_rel_error_ltp = max(max_rel_error_ltp, rel_diff_ltp)
        else:
            skipped_k.append(k)

        note = " *" if rec < rel_threshold else ""
        print(f"{k:>4}  {rec:>14.10f}  {ltp:>14.10f}  {sim:>14.10f}"
              f"  {abs_diff_rec:>11.2e}  {abs_diff_ltp:>11.2e}{note}")

    print("-" * 80)
    if skipped_k:
        print(f"  * k={skipped_k} excluded from rel-error check: P(S_N=k) < {rel_threshold:.4f}"
              f" (expected count < {MIN_EXPECTED_COUNT})")
    print(f"{'Total abs error':>22}  {'Recursive:':>10} {total_abs_error_rec:.6f}"
          f"  {'LTP:':>5} {total_abs_error_ltp:.6f}")
    print(f"{'Max rel error':>22}  {'Recursive:':>10} {max_rel_error_rec:.2%}"
          f"  {'LTP:':>5} {max_rel_error_ltp:.2%}"
          f"  (over reliable k only)")

    max_exact_diff = max(abs(recursive_probs[k] - ltp_probs[k]) for k in range(max_k + 1))
    print(f"{'Max |Rec - LTP|':>22}  {max_exact_diff:.2e}  (exact methods agree)")

    passed = max(max_rel_error_rec, max_rel_error_ltp) < 0.10
    print(f"\nConvergence check (max rel error < 10%, reliable k only): {'PASSED' if passed else 'FAILED'}")

    return passed


def run_simulation_comparisons():
    """Run all simulation vs recursive comparisons."""
    print("\n" + "=" * 70)
    print("SIMULATION VS RECURSIVE COMPUTATION COMPARISON")
    print("=" * 70)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    n_sims = 500000
    
    test_cases = [
        (
            "Poisson(λ=3)",
            [(DistributionType.POISSON, {"lambda": 3}, 1.0)]
        ),
        (
            "Negative Binomial(r=6, p=0.6)",
            [(DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.6}, 1.0)]
        ),
        (
            "Binomial(r=6, p=0.6)",
            [(DistributionType.BINOMIAL, {"r": 6, "p": 0.6}, 1.0)]
        ),
        (
            "Mixed Poisson (λ1=3, λ2=4; β=0.6, 0.4)",
            [
                (DistributionType.POISSON, {"lambda": 3}, 0.6),
                (DistributionType.POISSON, {"lambda": 4}, 0.4)
            ]
        ),
        (
            "Mixed Binomial (r1=8,p1=0.5; r2=10,p2=0.7; β=0.6,0.4)",
            [
                (DistributionType.BINOMIAL, {"r": 8, "p": 0.5}, 0.6),
                (DistributionType.BINOMIAL, {"r": 10, "p": 0.7}, 0.4)
            ]
        ),
        (
            "Mixed Negative Binomial (r1=4,p1=0.6; r2=6,p2=0.5; β=0.5,0.5)",
            [
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 4, "p": 0.6}, 0.5),
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.5}, 0.5)
            ]
        ),
        (
            "Mixed All Three (Poisson + Binomial + NegBin)",
            [
                (DistributionType.POISSON, {"lambda": 3}, 0.4),
                (DistributionType.BINOMIAL, {"r": 6, "p": 0.5}, 0.3),
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 2, "p": 0.4}, 0.3)
            ]
        ),
    ]
    
    results = []
    for name, components in test_cases:
        passed = compare_simulation_vs_recursive(
            name, components, claim_probs, max_k=10, n_simulations=n_sims
        )
        results.append((name, passed))
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPARISON SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} comparisons passed")
    print("=" * 70)
    
    return passed_count == total_count


# ============================================================================
# DISTRIBUTION COMPUTATION
# ============================================================================

def get_all_examples() -> Dict[str, Tuple[List[Tuple], dict]]:
    """
    Returns all thesis examples as a dictionary.

    Returns
    -------
    dict
        Mapping from example name to (components, claim_probs)
    """
    # Standard claim distribution from thesis
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}

    examples = {
        "Example 1: Poisson(λ=3)": (
            [(DistributionType.POISSON, {"lambda": 3}, 1.0)],
            claim_probs
        ),
        "Example 2: NegBin(r=6, p=0.6)": (
            [(DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.6}, 1.0)],
            claim_probs
        ),
        "Example 3: Binomial(r=6, p=0.6)": (
            [(DistributionType.BINOMIAL, {"r": 6, "p": 0.6}, 1.0)],
            claim_probs
        ),
        "Example 4: Mixed Poisson": (
            [
                (DistributionType.POISSON, {"lambda": 3}, 0.6),
                (DistributionType.POISSON, {"lambda": 4}, 0.4)
            ],
            claim_probs
        ),
        "Example 7: Mixed Binomial": (
            [
                (DistributionType.BINOMIAL, {"r": 8, "p": 0.5}, 0.6),
                (DistributionType.BINOMIAL, {"r": 10, "p": 0.7}, 0.4)
            ],
            claim_probs
        ),
        "Example 8: Mixed NegBin": (
            [
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 4, "p": 0.6}, 0.5),
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 6, "p": 0.5}, 0.5)
            ],
            claim_probs
        ),
        "Example 9: Mixed All Three": (
            [
                (DistributionType.POISSON, {"lambda": 3}, 0.4),
                (DistributionType.BINOMIAL, {"r": 6, "p": 0.5}, 0.3),
                (DistributionType.NEGATIVE_BINOMIAL, {"r": 2, "p": 0.4}, 0.3)
            ],
            claim_probs
        ),
    }

    return examples



# ============================================================================
# LAW OF TOTAL PROBABILITY METHOD
# ============================================================================

class LTPCompoundRV:
    """
    Computes P(S_N = k) for a compound random variable via the Law of Total Probability.

        P(S_N = k) = sum_{n=0}^{k} P(N = n) * P(X_1 + ... + X_n = k)

    The conditional probability P(X_1 + ... + X_n = k) is built up iteratively
    via n-fold convolution of the claim size distribution.

    Since P(X_i >= 1) = 1, exactly n claims cannot exceed k when n > k, so the
    sum is finite and exact.

    Parameters
    ----------
    components : List of tuples (dist_type, params, weight)
        Same format as MixedCompoundRV.
    claim_probs : dict
        P(X_1 = i) = α_i for i = 1, 2, 3, ...
    """

    def __init__(self, components: List[Tuple], claim_probs: dict):
        self.components = components
        self.claim_probs = claim_probs
        self._op_count = 0
        self._peak_memory = 0  # peak number of floats stored simultaneously in f

    def _pmf_N(self, n: int) -> float:
        """P(N = n) for the mixture at level s=0."""
        total = 0.0
        for dist_type, params, weight in self.components:
            pmf = 0.0
            if dist_type == DistributionType.POISSON:
                lam = params["lambda"]
                # Use log-space to avoid overflow for large n
                log_pmf = -lam + n * math.log(lam) - math.lgamma(n + 1)
                pmf = math.exp(log_pmf)
            elif dist_type == DistributionType.BINOMIAL:
                r, p = params["r"], params["p"]
                if 0 <= n <= r:
                    pmf = math.comb(r, n) * (p ** n) * ((1 - p) ** (r - n))
            elif dist_type == DistributionType.NEGATIVE_BINOMIAL:
                r, p = params["r"], params["p"]
                # Use log-space to avoid overflow for large n
                log_pmf = (math.lgamma(n + r) - math.lgamma(n + 1) - math.lgamma(r)
                           + r * math.log(p) + n * math.log(1 - p))
                pmf = math.exp(log_pmf)
            total += weight * pmf
        return total

    def compute_prob(self, k: int) -> float:
        """
        Compute P(S_N = k) via the Law of Total Probability.

        Iteratively builds the n-fold convolution f_n(j) = P(X_1+...+X_n = j)
        for n = 0, 1, ..., k and accumulates the weighted sum.

        Parameters
        ----------
        k : int
            The target value for S_N.

        Returns
        -------
        float
            P(S_N = k)
        """
        self._op_count = 0
        self._peak_memory = 0

        # f maps j -> P(X_1 + ... + X_n = j) for the current n.
        # Start with n=0: the empty sum equals 0 with probability 1.
        f = {0: 1.0}
        self._peak_memory = max(self._peak_memory, len(f))

        result = 0.0
        for n in range(0, k + 1):
            p_n = self._pmf_N(n)
            f_n_k = f.get(k, 0.0)
            result += p_n * f_n_k   # one multiply per n
            self._op_count += 1

            if n < k:
                # Advance: compute f_{n+1}(j) for j = 1..k
                f_new = {}
                for j in range(1, k + 1):
                    val = 0.0
                    for i, alpha_i in self.claim_probs.items():
                        prev = f.get(j - i, 0.0)
                        if prev != 0.0:
                            val += alpha_i * prev
                            self._op_count += 1  # one multiply per claim size term
                    if val != 0.0:
                        f_new[j] = val
                f = f_new
                self._peak_memory = max(self._peak_memory, len(f))

        return result

    def compute_distribution(self, max_k: int) -> dict:
        """
        Compute P(S_N = k) for k = 0, 1, ..., max_k in a single pass.

        Builds the n-fold convolution iteratively and accumulates the
        weighted sum P(N=n) * f_n(k) for every k simultaneously.

        Parameters
        ----------
        max_k : int
            Maximum value of k to compute.

        Returns
        -------
        dict
            Mapping from k to P(S_N = k).
        """
        self._op_count = 0
        self._peak_memory = 0

        result = {k: 0.0 for k in range(max_k + 1)}
        f = {0: 1.0}
        self._peak_memory = max(self._peak_memory, len(f))

        for n in range(0, max_k + 1):
            p_n = self._pmf_N(n)
            for j, f_val in f.items():
                result[j] += p_n * f_val
                self._op_count += 1

            if n < max_k:
                f_new = {}
                for j in range(1, max_k + 1):
                    val = 0.0
                    for i, alpha_i in self.claim_probs.items():
                        prev = f.get(j - i, 0.0)
                        if prev != 0.0:
                            val += alpha_i * prev
                            self._op_count += 1
                    if val != 0.0:
                        f_new[j] = val
                f = f_new
                self._peak_memory = max(self._peak_memory, len(f))

        return result

    def count_ops_for_k(self, k: int) -> int:
        """Return the operation count for computing P(S_N = k)."""
        self.compute_prob(k)
        return self._op_count

    def peak_memory_for_k(self, k: int) -> int:
        """Return the peak number of stored floats when computing P(S_N = k)."""
        self.compute_prob(k)
        return self._peak_memory


# ============================================================================
# INTUITIVE EXAMPLE AND COMPUTATION COUNT COMPARISON
# ============================================================================

def test_intuitive_example_ltp():
    """
    Reproduce the intuitive example from the thesis using the Law of Total Probability.

    N ~ Poisson(λ=3), claim distribution {1:0.05, 2:0.4, 3:0.1, 4:0.25, 5:0.2}.
    Expected: P(S_N = 4) = e^{-3} * 1.528521094
    """
    print("=" * 60)
    print("Intuitive Example: Law of Total Probability (LTP)")
    print("N ~ Poisson(λ=3)")
    print("=" * 60)

    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    components = [(DistributionType.POISSON, {"lambda": 3}, 1.0)]

    ltp_model = LTPCompoundRV(components, claim_probs)
    result = ltp_model.compute_prob(4)
    expected = 1.528521094 * math.exp(-3)

    print(f"Computed P(S_N = 4) = {result:.10f}")
    print(f"Expected P(S_N = 4) = {expected:.10f}")
    print(f"Difference:           {abs(result - expected):.2e}")
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-6)}\n")

    # Show the step-by-step breakdown matching the thesis
    print("Step-by-step LTP breakdown (conditioning on N):")
    lam = 3.0
    e_neg3 = math.exp(-3)
    for n in range(0, 5):
        p_n = math.exp(-lam) * (lam ** n) / math.factorial(n)
        # rebuild f_n(4) manually for display
        f = {0: 1.0}
        for _ in range(n):
            f_new = {}
            for j in range(1, 5):
                val = sum(claim_probs.get(i, 0) * f.get(j - i, 0)
                          for i in range(1, j + 1))
                if val != 0.0:
                    f_new[j] = val
            f = f_new
        f_n_4 = f.get(4, 0.0)
        term = p_n * f_n_4
        coeff = p_n / e_neg3
        print(f"  n={n}: P(N={n})={coeff:.6f}*e^(-3), "
              f"P(sum={4}|N={n})={f_n_4:.8f}, "
              f"term={term / e_neg3:.8f}*e^(-3)")

    return math.isclose(result, expected, rel_tol=1e-6)


def compare_computation_counts(name: str, components: List[Tuple],
                                claim_probs: dict,
                                k_values: List[int] = None):
    """
    Compare operations and peak memory between the recursive and LTP methods
    at various values of k for a single example.

    An 'operation' is one multiplication of a claim probability (α_i) by a
    previously-computed probability value in the core convolution sum.
    'Memory' is the peak number of probability values stored simultaneously:
    the recursive cache accumulates all (value, level) pairs; LTP keeps only
    the current convolution slice.

    Parameters
    ----------
    name : str
        Display name for this example.
    components : list
        Mixture components in MixedCompoundRV format.
    claim_probs : dict
        Claim size distribution.
    k_values : list of int
        Values of k to compare.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 20, 50, 100, 200]

    print("\n" + "=" * 95)
    print(f"COMPUTATION COUNT AND MEMORY: {name}")
    print("'Operations' = multiplications in the core convolution sum")
    print("'Memory'     = peak stored probability values  "
          "(Recursive: cumulative cache; LTP: one slice)")
    print("=" * 95)
    print(f"\n{'k':>5}  {'Rec Ops':>10}  {'LTP Ops':>10}  {'Op Ratio':>10}"
          f"  {'Rec Mem':>10}  {'LTP Mem':>10}  {'Mem Ratio':>10}")
    print("-" * 75)

    recursive_model = MixedCompoundRV(components, claim_probs)
    ltp_model = LTPCompoundRV(components, claim_probs)

    for k in k_values:
        rec_ops = recursive_model.count_ops_for_k(k)
        ltp_ops = ltp_model.count_ops_for_k(k)
        rec_mem = recursive_model.peak_memory_for_k(k)
        ltp_mem = ltp_model.peak_memory_for_k(k)
        op_ratio = ltp_ops / rec_ops if rec_ops > 0 else float("inf")
        mem_ratio = rec_mem / ltp_mem if ltp_mem > 0 else float("inf")
        print(f"{k:>5}  {rec_ops:>10,}  {ltp_ops:>10,}  {op_ratio:>9.2f}x"
              f"  {rec_mem:>10,}  {ltp_mem:>10,}  {mem_ratio:>9.2f}x")

    print("-" * 75)


def compare_all_examples_computation(k_values: List[int] = None):
    """
    Compare recursive vs. LTP computation counts and memory usage for every
    thesis example, first as a summary at k=4 (the thesis target), then as a
    detailed k-sweep for each individual example.

    Parameters
    ----------
    k_values : list of int
        Values of k for the detailed per-example sweep.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 20, 50, 100, 200]

    examples = get_all_examples()

    # ------------------------------------------------------------------ #
    # Summary table: all examples at k=4                                  #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 105)
    print("COMPUTATION COUNT AND MEMORY SUMMARY AT k=4: All Thesis Examples")
    print("Recursive vs. Law of Total Probability")
    print("'Operations' = multiplications in the core convolution sum")
    print("'Memory'     = peak stored values  "
          "(Recursive: cumulative cache; LTP: one slice)")
    print("=" * 105)
    print(f"\n{'Example':<35}  {'Rec Ops':>8}  {'LTP Ops':>8}  {'Op Ratio':>9}"
          f"  {'Rec Mem':>8}  {'LTP Mem':>8}  {'Mem Ratio':>9}")
    print("-" * 95)

    for name, (components, claim_probs) in examples.items():
        rec_model = MixedCompoundRV(components, claim_probs)
        ltp_model = LTPCompoundRV(components, claim_probs)
        rec_ops  = rec_model.count_ops_for_k(4)
        ltp_ops  = ltp_model.count_ops_for_k(4)
        rec_mem  = rec_model.peak_memory_for_k(4)
        ltp_mem  = ltp_model.peak_memory_for_k(4)
        op_ratio  = ltp_ops / rec_ops  if rec_ops  > 0 else float("inf")
        mem_ratio = rec_mem / ltp_mem  if ltp_mem  > 0 else float("inf")
        print(f"{name:<35}  {rec_ops:>8,}  {ltp_ops:>8,}  {op_ratio:>8.2f}x"
              f"  {rec_mem:>8,}  {ltp_mem:>8,}  {mem_ratio:>8.2f}x")

    print("-" * 95)
    print()
    print("Key finding: all examples produce identical operation and memory counts.")
    print("  The convolution-sum work depends only on the claim distribution support")
    print("  {1,2,3,4,5} and the target k — not on the distribution of N.  Whether")
    print("  N is Poisson, Binomial, Negative Binomial, or any mixture, the recursive")
    print("  method visits the same set of unique (value, level) pairs and the LTP")
    print("  method performs the same n-fold convolution steps.  The distribution of")
    print("  N affects only the scalar weights applied at each step, not the count.")
    print()
    print("Interpretation:")
    print("  Operations: Both methods are O(k²); the recursive method has a slight")
    print("    constant-factor advantage at small k via memoization, converging to")
    print("    the same cost as k grows.")
    print("  Memory: The recursive cache is O(k²) and never freed; LTP peak")
    print("    storage is O(k) — one convolution slice at a time.")

    # ------------------------------------------------------------------ #
    # Detailed k-sweep for each example                                   #
    # ------------------------------------------------------------------ #
    for name, (components, claim_probs) in examples.items():
        compare_computation_counts(name, components, claim_probs, k_values)


def compare_full_distribution_efficiency(max_k_values: List[int] = None):
    """
    Demonstrate the efficiency of computing the full distribution P(S_N = 0..K)
    in a single shared pass versus K+1 independent single-point queries.

    For the recursive method the shared pass keeps the memoization cache alive
    across all k values, so subproblems computed for smaller k are reused when
    computing larger k.  For LTP the shared pass builds the convolution table
    once and reads off all k values simultaneously.

    Both are contrasted against the cost of calling compute_prob(k) separately
    for each k from 0 to K, which discards all intermediate state between calls.

    Parameters
    ----------
    max_k_values : list of int
        Values of K (the upper limit of the distribution) to demonstrate.
    """
    if max_k_values is None:
        max_k_values = [5, 10, 20, 50, 100]

    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    components = [(DistributionType.POISSON, {"lambda": 3}, 1.0)]

    print("\n" + "=" * 95)
    print("FULL DISTRIBUTION EFFICIENCY: Single-Pass vs. Independent Per-k Queries")
    print("N ~ Poisson(λ=3), claim distribution support {1,2,3,4,5}")
    print("Computing P(S_N = k) for all k = 0, 1, ..., K")
    print("=" * 95)

    # Header
    print(f"\n{'':>4}  "
          f"{'--- Recursive ---':^38}  "
          f"{'--- LTP ---':^38}")
    print(f"{'K':>4}  "
          f"{'Independent':>12}  {'Shared pass':>12}  {'Savings':>11}  "
          f"{'Independent':>12}  {'Shared pass':>12}  {'Savings':>11}")
    print("-" * 93)

    rec = MixedCompoundRV(components, claim_probs)
    ltp = LTPCompoundRV(components, claim_probs)

    for K in max_k_values:
        # Independent queries: clear state before each k
        rec_independent = sum(rec.count_ops_for_k(k) for k in range(K + 1))
        ltp_independent = sum(ltp.count_ops_for_k(k) for k in range(K + 1))

        # Shared pass: one call computes all k = 0..K with shared state
        rec.clear_cache()
        rec._op_count = 0
        rec.compute_distribution(K)
        rec_shared = rec._op_count

        ltp._op_count = 0
        ltp.compute_distribution(K)
        ltp_shared = ltp._op_count

        rec_saving_pct = 100 * (rec_independent - rec_shared) / rec_independent
        ltp_saving_pct = 100 * (ltp_independent - ltp_shared) / ltp_independent

        print(f"{K:>4}  "
              f"{rec_independent:>12,}  {rec_shared:>12,}  {rec_saving_pct:>10.1f}%  "
              f"{ltp_independent:>12,}  {ltp_shared:>12,}  {ltp_saving_pct:>10.1f}%")

    print("-" * 93)
    print()
    print("Interpretation:")
    print("  Independent: compute_prob(k) called separately for each k = 0..K;")
    print("    the recursive cache is cleared and LTP rebuilds its convolution")
    print("    table from scratch on every call.")
    print("  Shared pass: compute_distribution(K) called once; the recursive")
    print("    cache accumulates across all k values so subproblems solved for")
    print("    smaller k are reused for larger k, and LTP builds the convolution")
    print("    table once and reads off every k simultaneously.")
    print("  Both methods achieve large savings with a shared pass.  The recursive")
    print("    savings grow because its cache depth (the level s) increases with k,")
    print("    creating more overlap between successive single-point queries.")



def compare_cached_vs_uncached(k_values: List[int] = None,
                                uncached_limit: int = 18):
    """
    Demonstrate the benefit of caching in the recursive method by comparing
    operation counts with and without memoization.

    Without caching the recursion recomputes the same (value, level)
    subproblems repeatedly, producing exponential growth.  The cache
    eliminates all duplicate work, reducing the cost to O(k²).

    Parameters
    ----------
    k_values : list of int
        Values of k to compare.
    uncached_limit : int
        Maximum k for which the uncached version is actually executed.
        Beyond this limit the growth is exponential and the runtime is
        infeasible; those rows display a note instead of a count.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 30, 50, 100]

    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    components = [(DistributionType.POISSON, {"lambda": 3}, 1.0)]
    model = MixedCompoundRV(components, claim_probs)

    print("\n" + "=" * 80)
    print("CACHING BENEFIT: Recursive Method With vs. Without Memoization")
    print("N ~ Poisson(λ=3), claim distribution support {1,2,3,4,5}")
    print("'Operations' = convolution-sum multiplications (α_i × probability)")
    print(f"Uncached ops computed directly for k ≤ {uncached_limit}; beyond that")
    print("  the exponential growth makes direct computation infeasible.")
    print("=" * 80)
    print(f"\n{'k':>5}  {'Cached':>12}  {'Uncached':>22}  {'Speedup':>18}  {'Cache entries':>14}")
    print("-" * 80)

    for k in k_values:
        cached_ops = model.count_ops_for_k(k)
        cache_entries = len(model._cache)

        if k <= uncached_limit:
            uncached_ops = model.count_ops_uncached(k)
            speedup = uncached_ops / cached_ops if cached_ops > 0 else float("inf")
            speedup_str = f"{speedup:,.1f}x"
            print(f"{k:>5}  {cached_ops:>12,}  {uncached_ops:>22,}  "
                  f"{speedup_str:>18}  {cache_entries:>14,}")
        else:
            print(f"{k:>5}  {cached_ops:>12,}  {'(too large to compute)':>22}  "
                  f"{'—':>18}  {cache_entries:>14,}")

    print("-" * 80)
    print()
    print("Interpretation:")
    print("  Each unique (value, level) pair can be reached by multiple paths")
    print("  through the recursion tree.  Without caching, every path recomputes")
    print("  the subproblem from scratch.  The number of such paths grows")
    print("  exponentially (~2^k), while the number of unique pairs grows as")
    print("  O(k²).  Caching converts the algorithm from exponential to")
    print("  polynomial time, at the cost of retaining all O(k²) cached values.")


if __name__ == "__main__":
    # Run recursive computation tests
    recursive_passed = run_all_tests()

    # Intuitive example via LTP
    print("\n\n")
    ltp_passed = test_intuitive_example_ltp()

    # Caching benefit comparison
    compare_cached_vs_uncached()

    # Computation count and memory comparison across all examples
    compare_all_examples_computation()

    # Full distribution efficiency: shared pass vs. independent queries
    compare_full_distribution_efficiency()

    # Run simulation comparisons
    print("\n\n")
    simulation_passed = run_simulation_comparisons()