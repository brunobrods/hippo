import random

import pytest

from coinbase.ga.ga_engine import (
    Elitism,
    FitnessEvaluation,
    GaConfig,
    GaConfigFile,
    GaussianMutation,
    Genome,
    GeneticAlgorithm,
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


def test_normalized_weights_falls_back_to_uniform_on_zero_sum():
    normalized = NormalizedWeights({"a": 0.0, "b": 0.0}).values()
    assert normalized == {"a": 0.5, "b": 0.5}


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
