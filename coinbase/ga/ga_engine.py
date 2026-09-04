import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


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
    # Off by default: every genome trained so far has non-negative weights, and
    # flipping this changes what the search can express, not just where it looks.
    allow_negative_weights: bool = False


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
            allow_negative_weights = bool(section.get("allow_negative_weights", False)),
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

    def has(self, key: str) -> bool:
        return key in self._weights

    def fitness(self) -> float:
        if self._fitness is None:
            raise ValueError(f"Genome has not been evaluated yet: {self._weights}")
        return self._fitness

    def is_evaluated(self) -> bool:
        return self._fitness is not None

    def scored(self, fitness: float) -> "Genome":
        return Genome(self._weights, fitness)


# A genome saved before a weight key existed, run against a config that now
# has it. The genome assigned that key no weight — it could not — so zero is
# the honest reading, and it leaves the score exactly what the genome was
# scored on when it was trained.
#
# Deliberately a decorator on the LOAD path rather than a softening of
# Genome.weight: training must still fail loudly on a key it does not know,
# because there a missing weight is a bug, not history.
class BackfilledGenome:
    def __init__(self, genome: Genome, keys: tuple[str, ...]) -> None:
        self._genome = genome
        self._keys   = keys

    def missing(self) -> tuple[str, ...]:
        return tuple(key for key in self._keys if not self._genome.has(key))

    def weights(self) -> dict[str, float]:
        return {**{key: 0.0 for key in self.missing()}, **self._genome.weights()}

    def filled(self) -> Genome:
        return Genome(self.weights())


# Scaled so the ABSOLUTE weights sum to 1, not the signed ones.
#
# Dividing by the signed sum breaks the moment weights may be negative: a
# genome whose weights nearly cancel has a sum near zero, and dividing by it
# explodes the weights without bound. The L1 norm has neither problem, and it
# is identical to the old behaviour whenever every weight is already positive
# — which is why enabling signed weights changes nothing for genomes that do
# not use them.
class NormalizedWeights:
    def __init__(self, raw: dict[str, float]) -> None:
        self._raw = raw

    def values(self) -> dict[str, float]:
        total = sum(abs(value) for value in self._raw.values())
        if total <= 0.0:
            share = 1.0 / len(self._raw)
            return {key: share for key in self._raw}
        return {key: value / total for key, value in self._raw.items()}


# How a genome's raw numbers are rescaled after every operator that produces
# new ones. This is a property of the MODEL, not of the search: L1 scaling
# means "one unit of conviction spread across the indicators", which is exactly
# what a weighted sum needs and exactly what a network's internal parameters
# must not have — scaling those would change the function the network computes.
#
# The engine therefore takes it rather than assuming it, and every operator
# defaults to L1 so the linear design and every test written against it behave
# as they always did.

class WeightScaling(Protocol):
    def scaled(self, raw: dict[str, float]) -> dict[str, float]: ...


class L1Scaling:
    def scaled(self, raw: dict[str, float]) -> dict[str, float]:
        return NormalizedWeights(raw).values()


class RandomWeights:
    def __init__(
        self,
        keys: tuple[str, ...],
        random_source: random.Random,
        allow_negative: bool = False,
        scaling: WeightScaling = L1Scaling(),
    ) -> None:
        self._keys           = keys
        self._random         = random_source
        self._allow_negative = allow_negative
        self._scaling        = scaling

    def generate(self) -> dict[str, float]:
        raw = {key: self._gene() for key in self._keys}
        return self._scaling.scaled(raw)

    # Sampled symmetrically about zero when signs are allowed, so the initial
    # population explores inverse relationships from the first generation
    # rather than having to mutate its way across zero.
    def _gene(self) -> float:
        if self._allow_negative:
            return self._random.uniform(-1.0, 1.0)
        return self._random.random()


class RandomPopulation:
    def __init__(
        self,
        size: int,
        keys: tuple[str, ...],
        random_source: random.Random,
        allow_negative: bool = False,
        scaling: WeightScaling = L1Scaling(),
    ) -> None:
        self._size           = size
        self._keys           = keys
        self._random         = random_source
        self._allow_negative = allow_negative
        self._scaling        = scaling

    def genomes(self) -> list[Genome]:
        random_weights = RandomWeights(
            self._keys, self._random, self._allow_negative, self._scaling,
        )
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
    def __init__(
        self,
        parent1: Genome,
        parent2: Genome,
        random_source: random.Random,
        scaling: WeightScaling = L1Scaling(),
    ) -> None:
        self._parent1 = parent1
        self._parent2 = parent2
        self._random  = random_source
        self._scaling = scaling

    def child(self) -> Genome:
        raw = {
            key: (self._parent1.weight(key) if self._random.random() < 0.5 else self._parent2.weight(key))
            for key in self._parent1.weights()
        }
        return Genome(self._scaling.scaled(raw))


class GaussianMutation:
    def __init__(
        self,
        genome: Genome,
        mutation_rate: float,
        sigma: float,
        random_source: random.Random,
        allow_negative: bool = False,
        scaling: WeightScaling = L1Scaling(),
    ) -> None:
        self._genome         = genome
        self._mutation_rate  = mutation_rate
        self._sigma          = sigma
        self._random         = random_source
        self._allow_negative = allow_negative
        self._scaling        = scaling

    def mutated(self) -> Genome:
        raw = {
            key: self._mutated_gene(value)
            for key, value in self._genome.weights().items()
        }
        return Genome(self._scaling.scaled(raw))

    # The clamp at zero is what made an inverse relationship inexpressible: a
    # genome could learn "RSI high means buy harder" but never "RSI high means
    # sell". Lifting it lets a weight cross zero and stay there.
    def _mutated_gene(self, value: float) -> float:
        if self._random.random() >= self._mutation_rate:
            return value
        mutated = value + self._random.gauss(0.0, self._sigma)
        return mutated if self._allow_negative else max(0.0, mutated)


class Elitism:
    def __init__(self, population: list[Genome], count: int) -> None:
        self._population = population
        self._count      = count

    def survivors(self) -> list[Genome]:
        return sorted(self._population, key=lambda genome: genome.fitness(), reverse=True)[: self._count]


class GenerationStats:
    def __init__(self, population: list[Genome]) -> None:
        self._population = population

    def best_fitness(self) -> float:
        return max(genome.fitness() for genome in self._population)

    def average_fitness(self) -> float:
        return sum(genome.fitness() for genome in self._population) / len(self._population)


# ── Engine ─────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    # `scaling` comes from the model design being trained — the engine evolves
    # numbers and does not know what they mean. It defaults to L1 so a caller
    # training the linear design need not say so.
    def __init__(
        self,
        config: GaConfig,
        keys: tuple[str, ...],
        scaling: WeightScaling = L1Scaling(),
    ) -> None:
        self._config  = config
        self._keys    = keys
        self._scaling = scaling
        self._random  = random.Random(config.seed)

    def initial_population(self) -> list[Genome]:
        return RandomPopulation(
            self._config.population_size, self._keys, self._random,
            self._config.allow_negative_weights, self._scaling,
        ).genomes()

    def evolve(
        self,
        fitness_function: FitnessFunction,
        on_generation: Optional[Callable[[int, float, float], None]] = None,
    ) -> Genome:
        population = FitnessEvaluation(self.initial_population(), fitness_function).scored()
        for generation in range(1, self._config.generations + 1):
            population = self._next_generation(population, fitness_function)
            if on_generation is not None:
                stats = GenerationStats(population)
                on_generation(generation, stats.best_fitness(), stats.average_fitness())
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
            candidate = UniformCrossover(parent1, parent2, self._random, self._scaling).child()
        else:
            candidate = parent1
        return GaussianMutation(
            candidate, self._config.mutation_rate, self._config.mutation_sigma,
            self._random, self._config.allow_negative_weights, self._scaling,
        ).mutated()
