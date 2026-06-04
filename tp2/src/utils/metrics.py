"""
Seguimiento y exportación de métricas por generación.

Usa pandas para registrar métricas de cada corrida,
comparar configuraciones y analizar experimentos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.genetic.individual import Individual


class MetricsTracker:
    """
    Registra métricas por generación usando pandas.

    Permite:
    - Registrar métricas por generación (fitness, error, tiempo)
    - Comparar configuraciones entre corridas
    - Exportar a CSV para análisis e informes
    - Obtener resúmenes estadísticos
    """

    def __init__(self, run_id: str, config_meta: Dict[str, Any]):
        """
        Args:
            run_id: Identificador único de la corrida.
            config_meta: Metadatos de configuración para la columna de comparación:
                selection, crossover, mutation, triangles, population, survival, etc.
        """
        self.run_id = run_id
        self.config_meta = config_meta
        self._rows: List[Dict[str, Any]] = []

    def record(self, generation: int, stats: Dict[str, Any], elapsed: float):
        """
        Registra las métricas de una generación.

        Args:
            generation: Número de generación.
            stats: Estadísticas de la población (best_fitness, avg_fitness, etc.).
            elapsed: Tiempo transcurrido en segundos desde el inicio de la corrida.
        """
        best_fitness = stats.get("best_fitness") or 0.0
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": stats.get("avg_fitness") or 0.0,
            "worst_fitness": stats.get("worst_fitness") or 0.0,
            # error = 1 - best_fitness (para fitness lineal, equivale a MSE normalizado)
            "error": round(1.0 - best_fitness, 6),
            "time_s": round(elapsed, 3),
            **self.config_meta,
        }
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna las métricas registradas como DataFrame de pandas."""
        return pd.DataFrame(self._rows)

    def export_csv(self, path: str | Path):
        """
        Exporta las métricas a un archivo CSV.

        Args:
            path: Ruta del archivo CSV de salida.
        """
        self.to_dataframe().to_csv(path, index=False)

    def summary(self) -> pd.DataFrame:
        """
        Retorna estadísticas descriptivas (count, mean, std, min, max) sobre
        las métricas principales de la corrida.

        Returns:
            DataFrame con describe() aplicado a best_fitness, avg_fitness,
            error y time_s.
        """
        df = self.to_dataframe()
        numeric_cols = [
            c
            for c in ["best_fitness", "avg_fitness", "error", "time_s"]
            if c in df.columns
        ]
        return df[numeric_cols].describe()

    @staticmethod
    def shapes_dataframe(
        individual: Individual, run_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Crea un DataFrame con una fila por forma del individuo."""
        rows = []
        for i, shape in enumerate(individual.triangles):
            row: Dict[str, Any] = {
                "order": i,
                "shape_type": individual.shape_type,
                "r": shape.color[0],
                "g": shape.color[1],
                "b": shape.color[2],
                "alpha": shape.color[3],
                "x0": None,
                "y0": None,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "cx": None,
                "cy": None,
                "rx": None,
                "ry": None,
                "angle": None,
            }

            if individual.shape_type == "triangle":
                row.update(
                    {
                        "x0": shape.vertices[0][0],
                        "y0": shape.vertices[0][1],
                        "x1": shape.vertices[1][0],
                        "y1": shape.vertices[1][1],
                        "x2": shape.vertices[2][0],
                        "y2": shape.vertices[2][1],
                    }
                )
            else:
                row.update(
                    {
                        "cx": shape.center[0],
                        "cy": shape.center[1],
                        "rx": shape.radii[0],
                        "ry": shape.radii[1],
                        "angle": shape.angle,
                    }
                )

            if run_id is not None:
                row["run_id"] = run_id
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def triangles_dataframe(
        individual: Individual, run_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crea un DataFrame con una fila por triángulo del individuo.

        Columnas: order, x0, y0, x1, y1, x2, y2, r, g, b, alpha[, run_id]

        Args:
            individual: Individuo con los triángulos a enumerar.
            run_id: Identificador de corrida (opcional, se añade como columna).

        Returns:
            DataFrame con los datos geométricos y de color de cada triángulo.
        """
        return MetricsTracker.shapes_dataframe(individual, run_id=run_id)

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """
        Carga métricas desde un CSV previamente exportado.

        Args:
            path: Ruta al archivo CSV.

        Returns:
            DataFrame con las métricas.
        """
        return pd.read_csv(path)

    @staticmethod
    def compare_runs(paths: List[str | Path]) -> pd.DataFrame:
        """
        Carga y concatena métricas de múltiples corridas para comparación.

        Útil para comparar distintas combinaciones de hiperparámetros.

        Args:
            paths: Lista de rutas a archivos CSV de métricas.

        Returns:
            DataFrame combinado con todas las corridas, listo para groupby/pivot.

        Example:
            >>> df = MetricsTracker.compare_runs(["run1/metrics.csv", "run2/metrics.csv"])
            >>> df.groupby("selection")["best_fitness"].max()
        """
        dfs = [pd.read_csv(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)
