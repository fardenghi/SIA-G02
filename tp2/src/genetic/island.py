"""
Modelo de Islas (IMGA) para el algoritmo genético.

Implementa un Island Model Genetic Algorithm donde K subpoblaciones
evolucionan en paralelo e intercambian individuos periódicamente
mediante migración en topología de anillo.

Referencia: "A Parallel Island Genetic Algorithm for Triangle-based
Image Reconstruction" (hal-05458101).
"""

from __future__ import annotations

import io
import multiprocessing as mp
import random
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from src.genetic.engine import (
    EvolutionResult,
    GeneticEngine,
    StepResult,
    create_engine,
)
from src.genetic.individual import Individual
from src.genetic.population import Population


# ---------------------------------------------------------------------------
# Topología
# ---------------------------------------------------------------------------

def build_ring_topology(num_islands: int) -> List[Tuple[int, int]]:
    """
    Construye una topología de anillo: isla i envía a isla (i+1) % K.

    Args:
        num_islands: Cantidad de islas (K).

    Returns:
        Lista de tuplas (source, destination).
    """
    return [(i, (i + 1) % num_islands) for i in range(num_islands)]


# ---------------------------------------------------------------------------
# Migración in-process (modo secuencial)
# ---------------------------------------------------------------------------

def migrate_sequential(
    engines: List[GeneticEngine],
    topology: List[Tuple[int, int]],
    migration_size: int,
    elite_count: int,
):
    """
    Ejecuta migración entre islas en el mismo proceso.

    Cada isla envía sus M mejores + M peores a su vecina en el anillo.
    Los migrantes reemplazan individuos aleatorios (no elite) en el destino.

    Args:
        engines: Lista de GeneticEngine, uno por isla.
        topology: Pares (source, destination).
        migration_size: M — cantidad de mejores y peores a migrar.
        elite_count: Individuos elite que no se reemplazan.
    """
    # Fase 1: recolectar emigrantes de cada fuente
    emigrants: Dict[int, List[Individual]] = {}
    for src_id, _dst_id in topology:
        if src_id in emigrants:
            continue
        pop = engines[src_id].population
        sorted_inds = sorted(
            pop.individuals, key=lambda ind: ind.fitness or 0.0, reverse=True
        )
        m = min(migration_size, len(sorted_inds) // 2)
        best_m = [ind.copy() for ind in sorted_inds[:m]]
        worst_m = [ind.copy() for ind in sorted_inds[-m:]]
        emigrants[src_id] = best_m + worst_m

    # Fase 2: inyectar inmigrantes en destinos
    for src_id, dst_id in topology:
        migrants = emigrants[src_id]
        if not migrants:
            continue

        dst_pop = engines[dst_id].population.individuals
        pop_size = len(dst_pop)

        # Indices reemplazables: excluir los top elite_count
        if elite_count > 0 and elite_count < pop_size:
            sorted_indices = sorted(
                range(pop_size),
                key=lambda i: dst_pop[i].fitness or 0.0,
                reverse=True,
            )
            elite_indices = set(sorted_indices[:elite_count])
            replaceable = [i for i in range(pop_size) if i not in elite_indices]
        else:
            replaceable = list(range(pop_size))

        n_replace = min(len(migrants), len(replaceable))
        chosen = random.sample(replaceable, n_replace)

        for i, idx in enumerate(chosen):
            migrant = migrants[i]
            migrant.fitness = None  # forzar re-evaluación en destino
            dst_pop[idx] = migrant


# ---------------------------------------------------------------------------
# Worker para multiprocessing
# ---------------------------------------------------------------------------

def _island_worker(
    island_id: int,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    target_image_bytes: bytes,
    engine_kwargs: dict,
):
    """
    Proceso worker para una isla.

    Crea su propio GeneticEngine y ejecuta comandos recibidos por queue.
    """
    # Reconstruir imagen desde bytes
    target_image = Image.open(io.BytesIO(target_image_bytes)).convert("RGB")

    # Crear engine en este proceso
    engine = create_engine(target_image=target_image, **engine_kwargs)

    while True:
        cmd = cmd_queue.get()
        if cmd is None or cmd[0] == "stop":
            break

        try:
            if cmd[0] == "init":
                engine.initialize_and_evaluate()
                best = engine.population.best
                result_queue.put((
                    "init_done",
                    island_id,
                    best.fitness,
                    engine.population.get_statistics(),
                ))

            elif cmd[0] == "step":
                # Guard: asegurar que la población tiene fitness evaluado
                # antes de cada step (defensa contra estado inconsistente
                # en procesos fork/spawn).
                if any(ind.fitness is None for ind in engine.population.individuals):
                    engine.evaluate_population(engine.population)
                step_result = engine.step()
                result_queue.put((
                    "step_done",
                    island_id,
                    {
                        "generation": step_result.generation,
                        "best_fitness": step_result.best_fitness,
                        "stats": step_result.stats,
                        "improved": step_result.improved,
                        "stopped_early": step_result.stopped_early,
                    },
                    engine.best_ever.to_dict(),
                ))

            elif cmd[0] == "get_emigrants":
                m = cmd[1]
                pop = engine.population
                sorted_inds = sorted(
                    pop.individuals,
                    key=lambda ind: ind.fitness or 0.0,
                    reverse=True,
                )
                m = min(m, len(sorted_inds) // 2)
                best_m = [ind.to_dict() for ind in sorted_inds[:m]]
                worst_m = [ind.to_dict() for ind in sorted_inds[-m:]]
                result_queue.put((
                    "emigrants",
                    island_id,
                    best_m + worst_m,
                ))

            elif cmd[0] == "inject_migrants":
                migrants_dicts = cmd[1]
                elite_count = cmd[2]
                pop = engine.population.individuals
                pop_size = len(pop)

                if elite_count > 0 and elite_count < pop_size:
                    sorted_indices = sorted(
                        range(pop_size),
                        key=lambda i: pop[i].fitness or 0.0,
                        reverse=True,
                    )
                    elite_indices = set(sorted_indices[:elite_count])
                    replaceable = [
                        i for i in range(pop_size) if i not in elite_indices
                    ]
                else:
                    replaceable = list(range(pop_size))

                n_replace = min(len(migrants_dicts), len(replaceable))
                chosen = random.sample(replaceable, n_replace)

                for i, idx in enumerate(chosen):
                    migrant = Individual.from_dict(migrants_dicts[i])
                    migrant.fitness = None
                    pop[idx] = migrant

                result_queue.put(("migrated", island_id))

        except Exception as e:
            tb = traceback.format_exc()
            result_queue.put(("error", island_id, f"{e}\n{tb}"))


# ---------------------------------------------------------------------------
# IslandEngine — orquestador principal
# ---------------------------------------------------------------------------

class IslandEngine:
    """
    Orquesta K GeneticEngine (islas) con migración en topología de anillo.

    Interfaz compatible con el uso en main.py: expone on_generation(),
    on_improvement(), run() -> EvolutionResult, y properties width/height.
    """

    def __init__(self, target_image: Image.Image, config):
        """
        Args:
            target_image: Imagen objetivo (PIL).
            config: Config completa (src.utils.config.Config).
        """
        self.config = config
        self.island_config = config.island
        self._target_image = target_image
        self._width, self._height = target_image.size

        # Serializar imagen para enviar a workers
        buf = io.BytesIO()
        target_image.save(buf, format="PNG")
        self._target_bytes = buf.getvalue()

        # Kwargs para create_engine (sin target_image)
        self._engine_kwargs = self._build_engine_kwargs()

        self._on_generation_callbacks: List[Callable] = []
        self._on_improvement_callbacks: List[Callable] = []

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def on_generation(self, callback: Callable):
        self._on_generation_callbacks.append(callback)

    def on_improvement(self, callback: Callable):
        self._on_improvement_callbacks.append(callback)

    def _build_engine_kwargs(self) -> dict:
        """Construye kwargs para create_engine a partir del Config."""
        c = self.config
        return dict(
            config=c.to_evolution_config(),
            selection_method=c.selection.method,
            tournament_size=c.selection.tournament_size,
            crossover_method=c.crossover.method,
            crossover_probability=c.crossover.probability,
            mutation_params=c.mutation.to_params(),
            threshold=c.selection.threshold,
            boltzmann_t0=c.selection.boltzmann_t0,
            boltzmann_tc=c.selection.boltzmann_tc,
            boltzmann_k=c.selection.boltzmann_k,
            survival_method=c.survival.method,
            survival_selection_method=c.survival.selection_method,
            offspring_ratio=c.survival.offspring_ratio,
            fitness_method=c.fitness.method,
            fitness_scale=c.fitness.exponential_scale,
            fitness_detail_weight_base=c.fitness.detail_weight_base,
            fitness_composite_alpha=c.fitness.composite_alpha,
            fitness_composite_beta=c.fitness.composite_beta,
            fitness_composite_gamma=c.fitness.composite_gamma,
            renderer=c.rendering.backend,
            adaptive_sigma=c.mutation.to_adaptive_sigma(),
        )

    def _notify_generation(self, gen: int, population: Population, stats: dict):
        for cb in self._on_generation_callbacks:
            cb(gen, population, stats)

    def _notify_improvement(self, gen: int, individual: Individual, fitness: float):
        for cb in self._on_improvement_callbacks:
            cb(gen, individual, fitness)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> EvolutionResult:
        if self.island_config.parallel:
            return self._run_parallel()
        else:
            return self._run_sequential()

    # ------------------------------------------------------------------
    # Modo secuencial
    # ------------------------------------------------------------------

    def _run_sequential(self) -> EvolutionResult:
        """Ejecuta islas secuencialmente en el mismo proceso."""
        start_time = time.time()
        K = self.island_config.num_islands
        M = self.island_config.migration_size
        T = self.island_config.migration_interval
        elite_count = self.config.survival.elite_count
        max_gen = self.config.max_generations
        topology = build_ring_topology(K)

        # Crear K engines
        engines: List[GeneticEngine] = []
        for _i in range(K):
            eng = create_engine(
                target_image=self._target_image, **self._engine_kwargs
            )
            engines.append(eng)

        # Inicializar y evaluar
        for eng in engines:
            eng.initialize_and_evaluate()

        # Tracking global
        global_best = max(
            (eng.best_ever for eng in engines), key=lambda ind: ind.fitness or 0.0
        )
        global_best = global_best.copy()
        global_best_fitness = global_best.fitness

        # Stats iniciales (gen 0) — agregar de la mejor isla
        best_engine = max(engines, key=lambda e: e.best_fitness_ever)
        self._notify_generation(0, best_engine.population, best_engine.history[0])

        history = [best_engine.history[0]]
        stopped_early = False

        for gen in range(1, max_gen + 1):
            # Migración
            if gen > 1 and T > 0 and gen % T == 0:
                migrate_sequential(engines, topology, M, elite_count)
                # Re-evaluar poblaciones con migrantes
                for eng in engines:
                    eng.evaluate_population(eng.population)

            # Step cada isla
            for eng in engines:
                eng.step()

            # Tracking global
            for eng in engines:
                if eng.best_fitness_ever > global_best_fitness:
                    global_best = eng.best_ever.copy()
                    global_best_fitness = eng.best_fitness_ever
                    self._notify_improvement(gen, global_best, global_best_fitness)

            # Stats globales
            all_fitnesses = [
                ind.fitness
                for eng in engines
                for ind in eng.population.individuals
                if ind.fitness is not None
            ]
            global_stats = {
                "generation": gen,
                "best_fitness": global_best_fitness,
                "worst_fitness": min(all_fitnesses) if all_fitnesses else 0.0,
                "avg_fitness": (
                    sum(all_fitnesses) / len(all_fitnesses)
                    if all_fitnesses
                    else 0.0
                ),
                "size": sum(len(eng.population) for eng in engines),
            }
            history.append(global_stats)

            # Usar la población de la mejor isla para el callback
            best_eng = max(engines, key=lambda e: e.best_fitness_ever)
            self._notify_generation(gen, best_eng.population, global_stats)

            # Early stopping
            if self.config.fitness_threshold is not None:
                if global_best_fitness >= self.config.fitness_threshold:
                    stopped_early = True
                    break

        elapsed = time.time() - start_time
        return EvolutionResult(
            best_individual=global_best,
            best_fitness=global_best_fitness,
            generations=gen if 'gen' in dir() else 0,
            elapsed_time=elapsed,
            history=history,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------------
    # Modo paralelo (multiprocessing)
    # ------------------------------------------------------------------

    def _run_parallel(self) -> EvolutionResult:
        """Ejecuta islas en procesos separados."""
        start_time = time.time()
        K = self.island_config.num_islands
        M = self.island_config.migration_size
        T = self.island_config.migration_interval
        elite_count = self.config.survival.elite_count
        max_gen = self.config.max_generations
        topology = build_ring_topology(K)

        # Crear queues y workers
        # Usar spawn siempre: crea procesos limpios sin heredar estado
        # del padre (necesario para GPU/OpenGL que no sobrevive fork).
        ctx = mp.get_context("spawn")
        cmd_queues: List[mp.Queue] = [ctx.Queue() for _ in range(K)]
        result_queue: mp.Queue = ctx.Queue()

        workers: List[mp.Process] = []
        for i in range(K):
            p = ctx.Process(
                target=_island_worker,
                args=(i, cmd_queues[i], result_queue, self._target_bytes, self._engine_kwargs),
                daemon=True,
            )
            p.start()
            workers.append(p)

        try:
            return self._parallel_loop(
                K, M, T, elite_count, max_gen, topology,
                cmd_queues, result_queue, workers, start_time,
            )
        finally:
            # Cleanup
            for i in range(K):
                try:
                    cmd_queues[i].put(("stop",))
                except Exception:
                    pass
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    def _parallel_loop(
        self,
        K: int,
        M: int,
        T: int,
        elite_count: int,
        max_gen: int,
        topology: List[Tuple[int, int]],
        cmd_queues: List[mp.Queue],
        result_queue: mp.Queue,
        workers: List[mp.Process],
        start_time: float,
    ) -> EvolutionResult:
        """Loop principal del modo paralelo."""

        # Inicializar todas las islas
        for i in range(K):
            cmd_queues[i].put(("init",))

        # Recolectar init results
        init_results = self._collect_results(result_queue, K, "init_done")
        best_init = max(init_results.values(), key=lambda r: r[1])
        global_best_fitness = best_init[1]
        global_best_dict = None  # se obtiene en el primer step
        # Placeholder individual para callbacks hasta tener el real
        global_best_individual = Individual.random(
            num_triangles=self.config.num_triangles,
            shape_type=self.config.shape_type,
        )
        global_best_individual.fitness = global_best_fitness

        # Stats gen 0
        gen0_stats = best_init[2]
        gen0_stats["generation"] = 0
        history = [gen0_stats]

        # Población proxy con el mejor global para callbacks (save image, etc.)
        proxy_pop = Population(individuals=[global_best_individual])
        self._notify_generation(0, proxy_pop, gen0_stats)

        stopped_early = False
        last_gen = 0

        for gen in range(1, max_gen + 1):
            last_gen = gen

            # Migración
            if gen > 1 and T > 0 and gen % T == 0:
                self._parallel_migrate(K, M, elite_count, topology, cmd_queues, result_queue)

            # Step todas las islas
            for i in range(K):
                cmd_queues[i].put(("step",))

            step_results = self._collect_results(result_queue, K, "step_done")

            # Tracking global
            improved = False
            for island_id, (_, step_data, best_dict) in step_results.items():
                if step_data["best_fitness"] > global_best_fitness:
                    global_best_fitness = step_data["best_fitness"]
                    global_best_dict = best_dict
                    improved = True

            if improved and global_best_dict is not None:
                global_best_individual = Individual.from_dict(global_best_dict)
                global_best_individual.fitness = global_best_fitness
                self._notify_improvement(gen, global_best_individual, global_best_fitness)

            # Stats globales
            island_stats = [r[1] for r in step_results.values()]
            avg_fitness = (
                sum(s["stats"].get("avg_fitness", 0) for s in island_stats) / K
            )
            worst_fitness = min(
                s["stats"].get("worst_fitness", 0) for s in island_stats
            )
            global_stats = {
                "generation": gen,
                "best_fitness": global_best_fitness,
                "worst_fitness": worst_fitness,
                "avg_fitness": avg_fitness,
                "size": sum(
                    s["stats"].get("size", 0) for s in island_stats
                ),
            }
            history.append(global_stats)
            # Población proxy con el mejor global para callbacks (save image)
            proxy_pop = Population(individuals=[global_best_individual])
            self._notify_generation(gen, proxy_pop, global_stats)

            # Early stopping
            if self.config.fitness_threshold is not None:
                if global_best_fitness >= self.config.fitness_threshold:
                    stopped_early = True
                    break

        elapsed = time.time() - start_time

        # Obtener el mejor individuo final
        if global_best_dict is not None:
            best_individual = Individual.from_dict(global_best_dict)
            best_individual.fitness = global_best_fitness
        else:
            # Fallback: pedir a la mejor isla
            best_individual = Individual.random(
                num_triangles=self.config.num_triangles
            )
            best_individual.fitness = global_best_fitness

        return EvolutionResult(
            best_individual=best_individual,
            best_fitness=global_best_fitness,
            generations=last_gen,
            elapsed_time=elapsed,
            history=history,
            stopped_early=stopped_early,
        )

    def _parallel_migrate(
        self,
        K: int,
        M: int,
        elite_count: int,
        topology: List[Tuple[int, int]],
        cmd_queues: List[mp.Queue],
        result_queue: mp.Queue,
    ):
        """Ejecuta migración entre islas en modo paralelo."""
        # Recolectar emigrantes de cada fuente
        sources = set(src for src, _dst in topology)
        for src_id in sources:
            cmd_queues[src_id].put(("get_emigrants", M))

        emigrant_results = self._collect_results(result_queue, len(sources), "emigrants")

        # Enviar migrantes a destinos
        for src_id, dst_id in topology:
            migrants_dicts = emigrant_results[src_id][1]
            cmd_queues[dst_id].put(("inject_migrants", migrants_dicts, elite_count))

        # Esperar confirmaciones
        self._collect_results(result_queue, K, "migrated")

    def _collect_results(
        self,
        result_queue: mp.Queue,
        expected: int,
        expected_type: str,
        timeout: float = 300.0,
    ) -> Dict[int, tuple]:
        """
        Recolecta `expected` resultados del tipo esperado desde la queue.

        Returns:
            Dict[island_id -> tuple con datos del resultado].
        """
        results: Dict[int, tuple] = {}
        deadline = time.time() + timeout
        while len(results) < expected:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timeout esperando {expected_type} de islas "
                    f"(recibidos {len(results)}/{expected})"
                )
            try:
                msg = result_queue.get(timeout=min(remaining, 10.0))
            except Exception:
                continue

            if msg[0] == "error":
                raise RuntimeError(
                    f"Error en isla {msg[1]}: {msg[2]}"
                )

            if msg[0] == expected_type:
                island_id = msg[1]
                results[island_id] = msg[1:]  # (island_id, ...)

        return results
