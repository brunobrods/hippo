import random
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from coinbase.ga.market_data_processor import NORMALIZED_COLUMNS

WEIGHT_KEYS = NORMALIZED_COLUMNS  # ("sma_short", "sma_long", "sma_extra", "rsi", "macd")


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GaConfig:
    population_size: int
    generations:     int
    mutation_rate:   float
    crossover_rate:  float
    tournament_size: int
    elitism_count:   int
    mutation_sigma:  float = 0.1
    seed:            Optional[int] = None


class GaConfigFile:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def config(self) -> GaConfig:
        section = self._raw["genetic_algorithm"]
        return GaConfig(
            population_size = section["population_size"],
            generations     = section["generations"],
            mutation_rate   = section["mutation_rate"],
            crossover_rate  = section["crossover_rate"],
            tournament_size = section["tournament_size"],
            elitism_count   = section["elitism_count"],
            mutation_sigma  = section.get("mutation_sigma", 0.1),
            seed            = section.get("seed"),
        )


# ── Genome ─────────────────────────────────────────────────────────────

class Genome:
    def __init__(self, weights: dict[str, float], fitness: Optional[float] = None) -> None:
        self._weights = weights
        self._fitness = fitness

    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def weight(self, key: str) -> float:
        return self._weights[key]

    def fitness(self) -> float:
        if self._fitness is None:
            raise ValueError(f"Genome has not been evaluated yet: {self._weights}")
        return self._fitness

    def is_evaluated(self) -> bool:
        return self._fitness is not None

    def scored(self, fitness: float) -> "Genome":
        return Genome(self._weights, fitness)


class NormalizedWeights:
    def __init__(self, raw: dict[str, float]) -> None:
        self._raw = raw

    def values(self) -> dict[str, float]:
        total = sum(self._raw.values())
        if total <= 0.0:
            share = 1.0 / len(self._raw)
            return {key: share for key in self._raw}
        return {key: value / total for key, value in self._raw.items()}


class RandomWeights:
    def __init__(self, keys: tuple[str, ...], random_source: random.Random) -> None:
        self._keys   = keys
        self._random = random_source

    def generate(self) -> dict[str, float]:
        raw = {key: self._random.random() for key in self._keys}
        return NormalizedWeights(raw).values()


class RandomPopulation:
    def __init__(self, size: int, keys: tuple[str, ...], random_source: random.Random) -> None:
        self._size   = size
        self._keys   = keys
        self._random = random_source

    def genomes(self) -> list[Genome]:
        random_weights = RandomWeights(self._keys, self._random)
        return [Genome(random_weights.generate()) for _ in range(self._size)]


# ── Fitness evaluation ─────────────────────────────────────────────────

class FitnessFunction(Protocol):
    def fitness(self, genome: Genome) -> float: ...


class FitnessEvaluation:
    def __init__(self, population: list[Genome], fitness_function: FitnessFunction) -> None:
        self._population       = population
        self._fitness_function = fitness_function

    def scored(self) -> list[Genome]:
        return [genome.scored(self._fitness_function.fitness(genome)) for genome in self._population]


# ── GA operators ───────────────────────────────────────────────────────

class TournamentSelection:
    def __init__(self, population: list[Genome], tournament_size: int, random_source: random.Random) -> None:
        self._population      = population
        self._tournament_size = tournament_size
        self._random          = random_source

    def winner(self) -> Genome:
        contestants = self._random.sample(self._population, self._tournament_size)
        return max(contestants, key=lambda genome: genome.fitness())


class UniformCrossover:
    def __init__(self, parent1: Genome, parent2: Genome, random_source: random.Random) -> None:
        self._parent1 = parent1
        self._parent2 = parent2
        self._random  = random_source

    def child(self) -> Genome:
        raw = {
            key: (self._parent1.weight(key) if self._random.random() < 0.5 else self._parent2.weight(key))
            for key in self._parent1.weights()
        }
        return Genome(NormalizedWeights(raw).values())


class GaussianMutation:
    def __init__(
        self,
        genome: Genome,
        mutation_rate: float,
        sigma: float,
        random_source: random.Random,
    ) -> None:
        self._genome        = genome
        self._mutation_rate = mutation_rate
        self._sigma         = sigma
        self._random        = random_source

    def mutated(self) -> Genome:
        raw = {
            key: self._mutated_gene(value)
            for key, value in self._genome.weights().items()
        }
        return Genome(NormalizedWeights(raw).values())

    def _mutated_gene(self, value: float) -> float:
        if self._random.random() >= self._mutation_rate:
            return value
        return max(0.0, value + self._random.gauss(0.0, self._sigma))


class Elitism:
    def __init__(self, population: list[Genome], count: int) -> None:
        self._population = population
        self._count      = count

    def survivors(self) -> list[Genome]:
        return sorted(self._population, key=lambda genome: genome.fitness(), reverse=True)[: self._count]


# ── Engine ─────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    def __init__(self, config: GaConfig, keys: tuple[str, ...] = WEIGHT_KEYS) -> None:
        self._config = config
        self._keys   = keys
        self._random = random.Random(config.seed)

    def initial_population(self) -> list[Genome]:
        return RandomPopulation(self._config.population_size, self._keys, self._random).genomes()

    def evolve(self, fitness_function: FitnessFunction) -> Genome:
        population = FitnessEvaluation(self.initial_population(), fitness_function).scored()
        for _ in range(self._config.generations):
            population = self._next_generation(population, fitness_function)
        return Elitism(population, 1).survivors()[0]

    def _next_generation(self, population: list[Genome], fitness_function: FitnessFunction) -> list[Genome]:
        elites   = Elitism(population, self._config.elitism_count).survivors()
        children = [self._child(population) for _ in range(self._config.population_size - len(elites))]
        return elites + FitnessEvaluation(children, fitness_function).scored()

    def _child(self, population: list[Genome]) -> Genome:
        selection = TournamentSelection(population, self._config.tournament_size, self._random)
        parent1   = selection.winner()
        parent2   = selection.winner()
        if self._random.random() < self._config.crossover_rate:
            candidate = UniformCrossover(parent1, parent2, self._random).child()
        else:
            candidate = parent1
        return GaussianMutation(candidate, self._config.mutation_rate, self._config.mutation_sigma, self._random).mutated()
