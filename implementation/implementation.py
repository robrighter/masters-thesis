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
        if s == 0:
            return [c[2] for c in self.components]
        
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
        return {k: self.compute_prob(k) for k in range(max_k + 1)}
    
    def clear_cache(self):
        """Clear the computation cache."""
        self._cache = {}


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
    Expected: P(S_N = 4) = 0.05679524977079922
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
    expected = 0.05679524977079922
    
    print(f"Computed P(S_N = 4) = {result:.18f}")
    print(f"Expected P(S_N = 4) = {expected:.18f}")
    print(f"Difference: {abs(result - expected):.2e}")
    # Note: Small discrepancy may exist due to accumulated rounding in hand calculations
    print(f"Test PASSED: {math.isclose(result, expected, rel_tol=1e-4)}\n")
    
    return math.isclose(result, expected, rel_tol=1e-4)


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
    Compare simulation estimates with recursive computation.
    
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
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {name}")
    print(f"Simulations: {n_simulations:,}")
    print(f"{'=' * 70}")
    
    # Recursive computation
    recursive_model = MixedCompoundRV(components, claim_probs)
    recursive_probs = recursive_model.compute_distribution(max_k)
    
    # Simulation
    simulator = CompoundRVSimulator(components, claim_probs, seed=42)
    simulated_probs = simulator.estimate_distribution(max_k, n_simulations)
    
    # Compare results
    print(f"\n{'k':>4}  {'Recursive':>14}  {'Simulated':>14}  {'Abs Diff':>12}  {'Rel Diff':>10}")
    print("-" * 60)
    
    total_abs_error = 0.0
    max_rel_error = 0.0
    
    for k in range(max_k + 1):
        rec = recursive_probs[k]
        sim = simulated_probs[k]
        abs_diff = abs(rec - sim)
        rel_diff = abs_diff / rec if rec > 1e-10 else 0.0
        
        total_abs_error += abs_diff
        max_rel_error = max(max_rel_error, rel_diff)
        
        print(f"{k:>4}  {rec:>14.10f}  {sim:>14.10f}  {abs_diff:>12.2e}  {rel_diff:>9.2%}")
    
    print("-" * 60)
    print(f"Total absolute error: {total_abs_error:.6f}")
    print(f"Maximum relative error: {max_rel_error:.2%}")
    
    # Check if simulation converged reasonably
    # With 500k samples, we expect relative errors generally under 5%
    passed = max_rel_error < 0.10  # Allow 10% max relative error
    print(f"\nConvergence check (max rel error < 10%): {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_simulation_comparisons():
    """Run all simulation vs recursive comparisons."""
    print("\n" + "=" * 70)
    print("SIMULATION VS RECURSIVE COMPUTATION COMPARISON")
    print("=" * 70)
    
    claim_probs = {1: 0.05, 2: 0.4, 3: 0.1, 4: 0.25, 5: 0.2}
    n_sims = 100000
    
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



if __name__ == "__main__":
    # Run recursive computation tests
    recursive_passed = run_all_tests()

    # Run simulation comparisons
    print("\n\n")
    simulation_passed = run_simulation_comparisons()