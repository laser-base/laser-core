import pytest
from scipy import stats

from laser.core.demographics.spatialpops import distribute_population_skewed
from laser.core.demographics.spatialpops import distribute_population_tapered
from laser.core.random import seed


# Test the basic functionality of distribute_population_skewed
# Given a total population of 100,000 across 5 nodes with 30% rural, when we
# distribute the population, then the sum, count, and urban share should match.
# Failure indicates the rural/urban split is broken or the sum is no longer conserved.
def test_distribute_population_skewed_basic():
    """Verify sum, node count, and urban-share constraints for the basic case."""
    seed(42)
    result = distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.3, min_pop=1000)
    assert sum(result) == 100_000  # Check total population is conserved
    assert len(result) == 5  # Verify the correct number of nodes
    assert result[0] == 70_000  # Urban node = (1 - frac_rural) * tot_pop


# Test the basic functionality of distribute_population_skewed with defaults
# Given the same scenario relying on default frac_rural and min_pop, when we
# distribute, then the same invariants must hold. Failure indicates a
# regression in the default-argument contract.
def test_distribute_population_skewed_basic_default():
    """Verify defaults (frac_rural=0.3, min_pop=1000) match the explicit case."""
    seed(42)
    result = distribute_population_skewed(tot_pop=100_000, num_nodes=5)
    assert sum(result) == 100_000
    assert len(result) == 5
    assert result[0] == 70_000


# Test an alternative scenario with distribute_population_skewed
# Given a population of 50,000 across 3 nodes with 40% rural, when we
# distribute, then total and node count must be correct and urban share
# matches 60% exactly. Failure indicates the urban fraction is miscomputed.
def test_distribute_population_skewed_alternative():
    """Verify scaling to a different (tot_pop, num_nodes, frac_rural) triple."""
    seed(42)
    result = distribute_population_skewed(tot_pop=50_000, num_nodes=3, frac_rural=0.4, min_pop=1000)
    assert sum(result) == 50_000  # Ensure total population matches input
    assert len(result) == 3  # Verify the correct number of nodes
    assert result[0] == 30_000  # Urban node = (1 - 0.4) * 50_000


# Test the per-node minimum floor
# Given a feasible (tot_pop, num_nodes, frac_rural, min_pop) combination,
# when we distribute, then every node — urban and rural — must be at least
# min_pop. Failure indicates the reserve-then-distribute construction is
# leaking values below the floor (e.g., via rounding).
def test_distribute_population_skewed_respects_min_pop():
    """Verify every node meets or exceeds min_pop across many random draws."""
    seed(42)
    for _ in range(50):
        result = distribute_population_skewed(tot_pop=100_000, num_nodes=10, frac_rural=0.5, min_pop=2000)
        assert (result >= 2000).all(), f"node below min_pop in {result.tolist()}"
        assert int(result.sum()) == 100_000


# Test the per-node minimum floor with min_pop=0
# Given min_pop=0, when we distribute, then the function should still produce
# a valid distribution (no minimum constraint). Failure indicates the
# code path for surplus_frac == frac_rural is broken.
def test_distribute_population_skewed_zero_min_pop():
    """Verify min_pop=0 disables the floor without breaking the sum/count."""
    seed(42)
    result = distribute_population_skewed(tot_pop=1000, num_nodes=5, frac_rural=0.3, min_pop=0)
    assert sum(result) == 1000
    assert len(result) == 5
    assert (result >= 0).all()


# Test reproducibility via laser.core.random.seed
# Given the same seed, when we call the function twice, then the outputs must
# be identical. Failure indicates the function is no longer using the seedable
# LASER PRNG (e.g., a stray np.random.* call slipped back in).
def test_distribute_population_skewed_reproducible():
    """Verify identical seed → identical output (project PRNG contract)."""
    seed(42)
    a = distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.3, min_pop=1000)
    seed(42)
    b = distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.3, min_pop=1000)
    assert (a == b).all()


# Test the shape of the rural distribution
# Given a large enough draw (1000 rural nodes) with min_pop=0 (so the floor
# does not distort the tail), when we distribute, then the rural node
# populations should be consistent with an exponential distribution under a
# Kolmogorov-Smirnov goodness-of-fit test. Failure indicates the underlying
# random draw is no longer exponential (e.g., a regression back to the prior
# 1/U / Pareto behavior, or a different RNG method).
def test_distribute_population_skewed_rural_is_exponential():
    """Rural populations pass a KS goodness-of-fit test against the exponential."""
    seed(42)
    tot_pop = 10_000_000
    num_nodes = 1001  # 1 urban + 1000 rural
    frac_rural = 0.3
    result = distribute_population_skewed(tot_pop=tot_pop, num_nodes=num_nodes, frac_rural=frac_rural, min_pop=0)

    assert int(result.sum()) == tot_pop
    assert len(result) == num_nodes
    # Urban node = (1 - frac_rural) * tot_pop plus the sub-unit rounding residual
    # absorbed across all 1000 rural rounds. Worst-case drift is num_rural/2.
    expected_urban = (1 - frac_rural) * tot_pop
    assert abs(int(result[0]) - expected_urban) <= (num_nodes - 1) / 2

    rural = result[1:].astype(float)
    assert len(rural) == 1000

    # MLE of an Exp(loc=0, scale=mean) fit. Estimating scale from the data makes
    # the KS p-value mildly anti-conservative, so we use a loose threshold; with
    # truly exponential samples the p-value at this scale is almost always >>0.05.
    ks_stat, p_value = stats.kstest(rural, "expon", args=(0.0, rural.mean()))
    assert p_value > 0.01, f"KS test rejected exponential fit: stat={ks_stat:.4f}, p={p_value:.4f}"


# Test edge case where number of nodes is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_skewed_zero_nodes():
    """Invalid num_nodes=0 must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Number of nodes must be greater than 0"):
        distribute_population_skewed(tot_pop=1000, num_nodes=0, frac_rural=0.3)


# Test edge case where total population is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_skewed_zero_population():
    """Invalid tot_pop=0 must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Total population must be greater than 0."):
        distribute_population_skewed(tot_pop=0, num_nodes=5, frac_rural=0.3)


# Test invalid rural fraction inputs for distribute_population_skewed
# Verifies the function raises errors for fractions outside the range [0, 1].
def test_distribute_population_skewed_invalid_fraction():
    """frac_rural outside [0, 1] must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(tot_pop=1000, num_nodes=5, frac_rural=-0.1, min_pop=0)
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(tot_pop=1000, num_nodes=5, frac_rural=1.5, min_pop=0)


# Test invalid min_pop input
# Given a negative min_pop, when we call the function, then a ValueError must
# be raised. Failure indicates the min_pop validation step is missing.
def test_distribute_population_skewed_negative_min_pop():
    """Negative min_pop must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Minimum node population must be non-negative."):
        distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.3, min_pop=-1)


# Test infeasible combination: urban node too small
# Given parameters where the urban share is below min_pop, when we call the
# function, then we must get a ValueError naming the urban constraint.
# Failure indicates the urban-feasibility check is missing or wrong.
def test_distribute_population_skewed_urban_infeasible():
    """Urban < min_pop must be detected and reported with a clear message."""
    # tot_pop=1000, frac_rural=0.3 ⇒ urban = 700, below min_pop=1000.
    with pytest.raises(ValueError, match="Urban node would receive"):
        distribute_population_skewed(tot_pop=1000, num_nodes=5, frac_rural=0.3, min_pop=1000)


# Test infeasible combination: rural budget too small
# Given parameters where the rural budget cannot fit num_rural * min_pop,
# when we call the function, then we must get a ValueError naming the rural
# constraint. This mirrors the "9 rural nodes × 1000 min, 6000 urban, 10k cap"
# scenario from the design discussion. Failure indicates the rural-feasibility
# check is missing.
def test_distribute_population_skewed_rural_infeasible():
    """Rural per-node average <= min_pop must be detected and reported."""
    # 9 rural nodes; rural budget = 0.4 * 10000 = 4000; per-node avg ≈ 444 < 1000.
    with pytest.raises(ValueError, match="Rural per-node average"):
        distribute_population_skewed(tot_pop=10_000, num_nodes=10, frac_rural=0.4, min_pop=1000)


# Test the rural feasibility boundary (tight budget is permitted)
# Given parameters where the rural per-node average exactly equals min_pop,
# when we distribute, then every rural node is pinned at exactly min_pop and
# the exponential surplus is zero. Failure indicates the feasibility check
# was tightened to a strict inequality or the surplus_frac == 0 branch is
# broken.
def test_distribute_population_skewed_rural_budget_tight():
    """Tight rural budget pins every rural node at exactly min_pop."""
    # 4 rural nodes × 1000 = 4000 = 0.4 * 10000 → per-node avg exactly 1000.
    seed(42)
    result = distribute_population_skewed(tot_pop=10_000, num_nodes=5, frac_rural=0.4, min_pop=1000)
    assert int(result.sum()) == 10_000
    assert int(result[0]) == 6000  # urban
    assert (result[1:] == 1000).all()  # every rural pinned at min_pop


# Test coupling: frac_rural=0 with multiple nodes is invalid
# Given frac_rural=0 and num_nodes > 1, when we call the function, then a
# ValueError must be raised — there is no rural budget but rural nodes were
# requested. Failure indicates the frac_rural ⟺ num_nodes coupling is missing.
def test_distribute_population_skewed_zero_frac_requires_single_node():
    """frac_rural=0 with num_nodes > 1 must be rejected."""
    with pytest.raises(ValueError, match="frac_rural=0 requires num_nodes=1"):
        distribute_population_skewed(tot_pop=100_000, num_nodes=5, frac_rural=0.0, min_pop=1000)


# Test coupling: frac_rural>0 with one node is invalid
# Given frac_rural > 0 and num_nodes = 1, when we call the function, then a
# ValueError must be raised — there is a rural budget but no rural node to
# receive it. Failure indicates the frac_rural ⟺ num_nodes coupling is missing.
def test_distribute_population_skewed_nonzero_frac_requires_multi_node():
    """frac_rural > 0 with num_nodes=1 must be rejected."""
    with pytest.raises(ValueError, match="requires num_nodes >= 2"):
        distribute_population_skewed(tot_pop=100_000, num_nodes=1, frac_rural=0.3, min_pop=1000)


# Test single-node degenerate case
# Given frac_rural=0 and num_nodes=1, when we distribute, then the sole urban
# node receives all of tot_pop. Failure indicates the single-node code path
# is broken or the coupling check is over-restrictive.
def test_distribute_population_skewed_single_node():
    """frac_rural=0 + num_nodes=1 returns the full population on the urban node."""
    result = distribute_population_skewed(tot_pop=100_000, num_nodes=1, frac_rural=0.0, min_pop=1000)
    assert len(result) == 1
    assert int(result[0]) == 100_000


# Test the basic functionality of distribute_population_tapered
# Verifies that the function correctly handles a population of 1000 across 5 nodes.
def test_distribute_population_tapered_basic():
    """Tapered distribution must preserve sum, count, and the decreasing pattern."""
    result = distribute_population_tapered(tot_pop=1000, num_nodes=5)
    assert sum(result) == 1000  # Ensure total population matches input
    assert len(result) == 5  # Verify the correct number of nodes
    assert result[0] > result[1]  # Check tapering pattern
    assert result[1] > result[2]


# Test edge case with small total population for distribute_population_tapered
# Ensures the function can handle small numbers without errors.
def test_distribute_population_tapered_small_population():
    """Small tot_pop with more nodes than agents must not crash."""
    result = distribute_population_tapered(tot_pop=10, num_nodes=4)
    assert sum(result) == 10  # Check total population matches input
    assert len(result) == 4  # Verify the correct number of nodes


# Test case where total population is evenly distributed among nodes
# Ensures the function can handle equal division correctly.
def test_distribute_population_tapered_equal_distribution():
    """num_nodes == tot_pop case still respects sum and count invariants."""
    result = distribute_population_tapered(tot_pop=10, num_nodes=10)
    assert sum(result) == 10  # Ensure total population matches input
    assert len(result) == 10  # Verify the correct number of nodes
    assert 0 in result  # Verify some nodes may have zero population


# Test case with a large number of nodes relative to population
# Ensures the function handles large node counts gracefully.
def test_distribute_population_tapered_large_nodes():
    """Very many nodes per agent must preserve sum/count and the taper."""
    result = distribute_population_tapered(tot_pop=100, num_nodes=50)
    assert sum(result) == 100  # Ensure total population matches input
    assert len(result) == 50  # Verify the correct number of nodes
    assert result[0] > result[-1]  # Check tapering pattern


# Test edge case where number of nodes is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_tapered_zero_nodes():
    """Invalid num_nodes=0 must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Both tot_pop and num_nodes must be greater than 0."):
        distribute_population_tapered(tot_pop=1000, num_nodes=0)


# Test edge case where total population is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_tapered_zero_population():
    """Invalid tot_pop=0 must be rejected at the API boundary."""
    with pytest.raises(ValueError, match="Both tot_pop and num_nodes must be greater than 0."):
        distribute_population_tapered(tot_pop=0, num_nodes=5)


# Test adjustment logic in distribute_population_tapered
# Verifies that the function correctly adjusts the population to match input.
def test_distribute_population_tapered_adjustment():
    """Rounding-residual absorption must keep the sum exactly equal to tot_pop."""
    result = distribute_population_tapered(tot_pop=1200, num_nodes=3)
    assert sum(result) == 1200  # Ensure total population matches input
    assert len(result) == 3  # Verify the correct number of nodes
