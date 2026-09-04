import random

import pytest

from coinbase.ga.ga_engine import (
    BackfilledGenome,
    Elitism,
    FitnessEvaluation,
    GaConfig,
    GaConfigFile,
    GaussianMutation,
    Genome,
    GeneticAlgorithm,
    L1Scaling,
    NormalizedWeights,
    RandomPopulation,
    RandomWeights,
    TournamentSelection,
    UniformCrossover,
)

_KEYS = ("sma_short", "sma_long", "sma_extra", "rsi", "macd")


# ── Genome ─────────────────────────────────────────────────────────────

def test_genome_weights_returns_a_copy():
    genome  = Genome({"rsi": 1.0})
    weights = genome.weights()
    weights["rsi"] = 0.0
    assert genome.weight("rsi") == 1.0


def test_genome_fitness_raises_before_evaluation():
    with pytest.raises(ValueError):
        Genome({"rsi": 1.0}).fitness()


def test_genome_scored_returns_new_instance_leaving_original_unevaluated():
    genome = Genome({"rsi": 1.0})
    scored = genome.scored(4.2)
    assert not genome.is_evaluated()
    assert scored.is_evaluated()
    assert scored.fitness() == pytest.approx(4.2)


# ── NormalizedWeights / RandomWeights / RandomPopulation ───────────────

def test_normalized_weights_sum_to_one():
    normalized = NormalizedWeights({"a": 1.0, "b": 3.0}).values()
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["b"] == pytest.approx(0.75)


# The L1 norm is what makes signed weights safe: dividing by the SIGNED sum
# explodes when weights nearly cancel.
def test_normalized_weights_scale_by_absolute_mass():
    normalized = NormalizedWeights({"a": -3.0, "b": 1.0}).values()
    assert sum(abs(v) for v in normalized.values()) == pytest.approx(1.0)
    assert normalized["a"] == pytest.approx(-0.75)


def test_nearly_cancelling_weights_do_not_explode():
    normalized = NormalizedWeights({"a": 1.0, "b": -0.999999}).values()
    assert all(abs(v) <= 1.0 for v in normalized.values())


def test_normalizing_is_unchanged_for_all_positive_weights():
    # The backward-compatibility guarantee: every genome trained before signed
    # weights existed must normalize to exactly what it always did.
    raw = {"a": 2.0, "b": 3.0, "c": 5.0}
    assert NormalizedWeights(raw).values() == pytest.approx({"a": 0.2, "b": 0.3, "c": 0.5})


def test_normalized_weights_falls_back_to_uniform_on_zero_sum():
    normalized = NormalizedWeights({"a": 0.0, "b": 0.0}).values()
    assert normalized == {"a": 0.5, "b": 0.5}


# ── WeightScaling ────────────────────────────────────────────────────

def test_l1_scaling_is_the_normalization_the_engine_has_always_applied():
    raw = {"a": 2.0, "b": -3.0, "c": 5.0}
    assert L1Scaling().scaled(raw) == pytest.approx(NormalizedWeights(raw).values())


def test_every_operator_takes_the_scaling_rather_than_assuming_l1():
    # The seam that lets a design whose parameters must NOT be rescaled — a
    # network's internals — be evolved by the same engine. Identity scaling is
    # not a design this build ships; it stands in here for any non-linear one.
    class _Identity:
        def scaled(self, raw: dict[str, float]) -> dict[str, float]:
            return dict(raw)

    genome  = Genome({"a": 2.0, "b": 6.0})
    mutated = GaussianMutation(
        genome, mutation_rate=0.0, sigma=0.5,
        random_source=random.Random(1), scaling=_Identity(),
    ).mutated()
    # Untouched by mutation AND untouched by scaling: still the raw magnitudes.
    assert mutated.weights() == pytest.approx({"a": 2.0, "b": 6.0})

    child = UniformCrossover(genome, genome, random.Random(1), scaling=_Identity()).child()
    assert child.weights() == pytest.approx({"a": 2.0, "b": 6.0})

    weights = RandomWeights(("a", "b"), random.Random(1), scaling=_Identity()).generate()
    assert sum(abs(value) for value in weights.values()) != pytest.approx(1.0)


def test_random_weights_generate_sum_to_one():
    weights = RandomWeights(_KEYS, random.Random(1)).generate()
    assert set(weights) == set(_KEYS)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_random_population_produces_normalized_genomes_of_requested_size():
    genomes = RandomPopulation(10, _KEYS, random.Random(2)).genomes()
    assert len(genomes) == 10
    for genome in genomes:
        assert sum(genome.weights().values()) == pytest.approx(1.0)


# ── FitnessEvaluation ────────────────────────────────────────────────

class _WeightSumFitness:
    def __init__(self, key: str) -> None:
        self._key = key

    def fitness(self, genome: Genome) -> float:
        return genome.weight(self._key)


def test_fitness_evaluation_scores_every_genome():
    population = RandomPopulation(5, _KEYS, random.Random(3)).genomes()
    scored     = FitnessEvaluation(population, _WeightSumFitness("rsi")).scored()
    assert all(genome.is_evaluated() for genome in scored)
    assert [g.fitness() for g in scored] == [g.weight("rsi") for g in scored]


# ── TournamentSelection ──────────────────────────────────────────────

def test_tournament_selection_picks_the_fittest_contestant():
    population = [
        Genome({"rsi": 1.0}, fitness=1.0),
        Genome({"rsi": 1.0}, fitness=5.0),
        Genome({"rsi": 1.0}, fitness=2.0),
    ]
    winner = TournamentSelection(population, tournament_size=3, random_source=random.Random(0)).winner()
    assert winner.fitness() == 5.0


# ── UniformCrossover ─────────────────────────────────────────────────

def test_uniform_crossover_child_is_normalized_and_uses_only_parent_keys():
    parent1 = Genome({"a": 1.0, "b": 0.0})
    parent2 = Genome({"a": 0.0, "b": 1.0})
    child   = UniformCrossover(parent1, parent2, random.Random(7)).child()
    assert set(child.weights()) == {"a", "b"}
    assert sum(child.weights().values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in child.weights().values())


def test_uniform_crossover_varies_the_child_across_seeds():
    parent1 = Genome({"a": 0.9, "b": 0.1})
    parent2 = Genome({"a": 0.1, "b": 0.9})
    children_a = {
        round(UniformCrossover(parent1, parent2, random.Random(seed)).child().weight("a"), 6)
        for seed in range(20)
    }
    assert len(children_a) > 1  # recombination actually mixes genes, not always cloning one parent


# ── GaussianMutation ─────────────────────────────────────────────────

def test_gaussian_mutation_always_renormalizes():
    genome  = Genome({"a": 0.5, "b": 0.5})
    mutated = GaussianMutation(genome, mutation_rate=1.0, sigma=0.5, random_source=random.Random(9)).mutated()
    assert sum(mutated.weights().values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in mutated.weights().values())


# The clamp at zero is exactly what made an inverse relationship
# inexpressible: a genome could learn "RSI high -> buy harder" but never
# "RSI high -> sell".
def test_mutation_clamps_at_zero_by_default():
    genome  = Genome({"a": 0.02, "b": 0.98})
    for seed in range(30):
        mutated = GaussianMutation(genome, 1.0, 0.5, random.Random(seed)).mutated()
        assert all(value >= 0.0 for value in mutated.weights().values())


def test_mutation_can_cross_zero_when_signs_are_allowed():
    genome   = Genome({"a": 0.02, "b": 0.98})
    negative = [
        GaussianMutation(genome, 1.0, 0.5, random.Random(seed), allow_negative=True).mutated()
        for seed in range(30)
    ]
    assert any(v < 0.0 for m in negative for v in m.weights().values())
    # Still L1-normalized, whatever the signs.
    for m in negative:
        assert sum(abs(v) for v in m.weights().values()) == pytest.approx(1.0)


def test_a_signed_initial_population_explores_both_directions():
    plain  = RandomPopulation(50, _KEYS, random.Random(1)).genomes()
    signed = RandomPopulation(50, _KEYS, random.Random(1), allow_negative=True).genomes()
    assert all(v >= 0.0 for g in plain for v in g.weights().values())
    assert any(v < 0.0 for g in signed for v in g.weights().values())


def test_the_ga_config_flag_reaches_the_search():
    config = GaConfig(
        population_size=20, generations=3, mutation_rate=1.0, crossover_rate=0.8,
        tournament_size=3, elitism_count=1, mutation_sigma=0.4, seed=5,
        allow_negative_weights=True,
    )
    best = GeneticAlgorithm(config, _KEYS).evolve(_WeightSumFitness("rsi"))
    assert sum(abs(v) for v in best.weights().values()) == pytest.approx(1.0)


def test_gaussian_mutation_rate_zero_leaves_weights_unchanged_after_renormalization():
    genome  = Genome({"a": 0.5, "b": 0.5})
    mutated = GaussianMutation(genome, mutation_rate=0.0, sigma=0.5, random_source=random.Random(9)).mutated()
    assert mutated.weights() == pytest.approx(genome.weights())


# ── Elitism ──────────────────────────────────────────────────────────

def test_elitism_keeps_top_n_by_fitness():
    population = [
        Genome({}, fitness=3.0),
        Genome({}, fitness=1.0),
        Genome({}, fitness=5.0),
        Genome({}, fitness=2.0),
    ]
    survivors = Elitism(population, count=2).survivors()
    assert [g.fitness() for g in survivors] == [5.0, 3.0]


# ── GaConfigFile ─────────────────────────────────────────────────────

def test_ga_config_file_reads_section_with_defaults():
    raw = {
        "genetic_algorithm": {
            "population_size": 20,
            "generations": 5,
            "mutation_rate": 0.2,
            "crossover_rate": 0.7,
            "tournament_size": 3,
            "elitism_count": 2,
        }
    }
    config = GaConfigFile(raw).config()
    assert config.population_size == 20
    assert config.mutation_sigma == pytest.approx(0.1)
    assert config.seed is None


# ── GeneticAlgorithm.evolve ──────────────────────────────────────────

def test_evolve_drives_weight_toward_what_fitness_rewards():
    config = GaConfig(
        population_size=40,
        generations=25,
        mutation_rate=0.3,
        crossover_rate=0.8,
        tournament_size=4,
        elitism_count=4,
        mutation_sigma=0.15,
        seed=42,
    )
    best = GeneticAlgorithm(config, _KEYS).evolve(_WeightSumFitness("rsi"))
    assert best.is_evaluated()
    assert best.weight("rsi") > 1.0 / len(_KEYS)  # better than the uniform-random baseline


def test_evolve_invokes_on_generation_once_per_generation_with_plausible_stats():
    config = GaConfig(
        population_size=10,
        generations=4,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=3,
        elitism_count=2,
        seed=7,
    )
    calls: list[tuple[int, float, float]] = []
    GeneticAlgorithm(config, _KEYS).evolve(_WeightSumFitness("rsi"), on_generation=lambda g, b, a: calls.append((g, b, a)))

    assert [generation for generation, _, _ in calls] == [1, 2, 3, 4]
    for _, best, average in calls:
        assert best >= average  # best-of-population can never be below the population average


def test_evolve_is_reproducible_with_a_fixed_seed():
    config = GaConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=3,
        elitism_count=2,
        seed=123,
    )
    fitness_function = _WeightSumFitness("macd")
    best_a = GeneticAlgorithm(config, _KEYS).evolve(fitness_function)
    best_b = GeneticAlgorithm(config, _KEYS).evolve(fitness_function)
    assert best_a.weights() == pytest.approx(best_b.weights())
    assert best_a.fitness() == pytest.approx(best_b.fitness())


# ── BackfilledGenome ───────────────────────────────────────────────────
# A genome saved before a weight key existed, run against a config that now
# has it. Genome.weight raises on a missing key, so without this every one of
# the genomes trained before index_z would fail the moment it was added.

def test_a_genome_missing_a_new_key_reports_it():
    backfilled = BackfilledGenome(Genome({"rsi": 0.6, "macd": 0.4}), ("rsi", "macd", "index_z"))
    assert backfilled.missing() == ("index_z",)


def test_a_missing_key_is_backfilled_at_zero():
    # Zero is the honest reading: the genome could not have used the key, so
    # the score stays exactly what it was scored on during training.
    filled = BackfilledGenome(Genome({"rsi": 0.6, "macd": 0.4}), ("rsi", "macd", "index_z")).filled()
    assert filled.weight("index_z") == 0.0
    assert filled.weight("rsi") == 0.6


def test_a_complete_genome_is_left_alone():
    genome     = Genome({"rsi": 0.6, "macd": 0.4})
    backfilled = BackfilledGenome(genome, ("rsi", "macd"))
    assert backfilled.missing() == ()
    assert backfilled.filled().weights() == genome.weights()


def test_backfilling_never_overwrites_a_weight_the_genome_has():
    filled = BackfilledGenome(Genome({"rsi": 0.9, "index_z": 0.1}), ("rsi", "index_z")).filled()
    assert filled.weight("index_z") == 0.1


def test_a_genome_knows_which_keys_it_carries():
    genome = Genome({"rsi": 1.0})
    assert genome.has("rsi") is True
    assert genome.has("index_z") is False
