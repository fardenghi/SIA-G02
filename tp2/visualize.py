"""
Visualizador gráfico de la evolución del algoritmo genético.

Permite explorar paso a paso la secuencia de generaciones,
mostrando el individuo renderizado y métricas relevantes.

Uso:
    python visualize.py output/mi_experimento
    python visualize.py output/mi_experimento --export-gif evolucion.gif
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image


class EvolutionVisualizer:
    """
    Visualizador interactivo de la evolución del algoritmo genético.

    Carga los resultados de un directorio de salida y permite
    navegar entre generaciones con controles interactivos.
    """

    def __init__(self, output_dir: str | Path):
        """
        Args:
            output_dir: Directorio con los resultados del algoritmo.
        """
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.output_dir}")

        # Cargar datos
        self.frames = self._load_frames()
        self.history = self._load_history()
        self.triangles_data = self._load_triangles()
        self.target_image = self._load_target()

        if not self.frames:
            raise ValueError(
                "No se encontraron imágenes de generaciones en el directorio"
            )

        self.current_frame = 0
        self.fig = None
        self.ax_main = None
        self.ax_info = None
        self.ax_plot = None

    def _load_frames(self) -> List[Tuple[int, Path]]:
        """Carga las rutas de los frames ordenados por generación."""
        frames = []

        # Buscar imágenes de generación
        for img_path in sorted(self.output_dir.glob("gen_*.png")):
            # Extraer número de generación del nombre
            gen_str = img_path.stem.replace("gen_", "")
            try:
                gen = int(gen_str)
                frames.append((gen, img_path))
            except ValueError:
                continue

        # Agregar resultado final
        result_path = self.output_dir / "result.png"
        if result_path.exists():
            if frames:
                # El resultado final corresponde a la última generación
                last_gen = frames[-1][0] if frames else 0
                frames.append((last_gen, result_path))
            else:
                frames.append((0, result_path))

        return frames

    def _load_history(self) -> List[Dict]:
        """Carga el historial de métricas si existe."""
        # Intentar cargar desde CSV
        csv_path = self.output_dir / "metrics.csv"
        if csv_path.exists():
            import csv

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                return [
                    {
                        k: float(v) if v and k != "generation" else (int(v) if v else 0)
                        for k, v in row.items()
                    }
                    for row in reader
                ]

        # Si no hay CSV, intentar reconstruir desde los frames
        return []

    def _load_triangles(self) -> Optional[Dict]:
        """Carga los datos de triángulos si existen."""
        json_path = self.output_dir / "triangles.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        return None

    def _load_target(self) -> Optional[np.ndarray]:
        """Intenta cargar la imagen objetivo si existe."""
        # Buscar en el directorio padre o en input/
        possible_paths = [
            self.output_dir / "target.png",
            self.output_dir.parent / "input" / "*.png",
            self.output_dir.parent / "input" / "*.jpg",
        ]

        for pattern in possible_paths:
            if "*" in str(pattern):
                matches = list(Path(pattern.parent).glob(pattern.name))
                if matches:
                    return np.array(Image.open(matches[0]).convert("RGB"))
            elif pattern.exists():
                return np.array(Image.open(pattern).convert("RGB"))

        return None

    def _get_frame_data(self, frame_idx: int) -> Tuple[int, np.ndarray, Dict]:
        """
        Obtiene los datos de un frame específico.

        Returns:
            Tupla (generación, imagen, métricas).
        """
        gen, img_path = self.frames[frame_idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Buscar métricas correspondientes
        metrics = {}
        if self.history:
            # Buscar la entrada más cercana
            for h in self.history:
                if h.get("generation", 0) <= gen:
                    metrics = h
                else:
                    break

        return gen, image, metrics

    def _format_metrics(self, gen: int, metrics: Dict) -> str:
        """Formatea las métricas para mostrar."""
        lines = [
            f"GENERACIÓN: {gen}",
            "-" * 25,
        ]

        if metrics:
            if "best_fitness" in metrics:
                lines.append(f"Mejor Fitness: {metrics['best_fitness']:.2f}")
            if "avg_fitness" in metrics:
                lines.append(f"Fitness Promedio: {metrics['avg_fitness']:.2f}")
            if "worst_fitness" in metrics:
                lines.append(f"Peor Fitness: {metrics['worst_fitness']:.2f}")

        if self.triangles_data:
            lines.append("-" * 25)
            lines.append(
                f"Triángulos: {self.triangles_data.get('num_triangles', 'N/A')}"
            )

        lines.append("-" * 25)
        lines.append(f"Frame: {self.current_frame + 1} / {len(self.frames)}")

        return "\n".join(lines)

    def show(self):
        """Muestra el visualizador interactivo."""
        # Crear figura con layout personalizado
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(
            "Evolución del Algoritmo Genético", fontsize=14, fontweight="bold"
        )

        # Crear grid: imagen principal (grande), info (derecha), gráfico (abajo)
        gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[0.1, 2, 0.8],
            height_ratios=[2, 0.15, 0.6],
            hspace=0.3,
            wspace=0.2,
        )

        # Área principal para la imagen
        self.ax_main = self.fig.add_subplot(gs[0, 1])
        self.ax_main.set_title("Individuo Actual")
        self.ax_main.axis("off")

        # Área de información
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_info.axis("off")

        # Área para gráfico de fitness
        self.ax_plot = self.fig.add_subplot(gs[2, :])
        self.ax_plot.set_xlabel("Generación")
        self.ax_plot.set_ylabel("Fitness (MSE)")
        self.ax_plot.set_title("Evolución del Fitness")
        self.ax_plot.grid(True, alpha=0.3)

        # Slider para navegación
        ax_slider = self.fig.add_subplot(gs[1, 1])
        self.slider = Slider(
            ax_slider, "Generación", 0, len(self.frames) - 1, valinit=0, valstep=1
        )
        self.slider.on_changed(self._on_slider_change)

        # Botones de navegación
        ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.65, 0.02, 0.1, 0.04])
        ax_play = plt.axes([0.45, 0.02, 0.1, 0.04])

        self.btn_prev = Button(ax_prev, "◀ Anterior")
        self.btn_next = Button(ax_next, "Siguiente ▶")
        self.btn_play = Button(ax_play, "▶ Play")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_play.on_clicked(self._on_play)

        # Estado de reproducción
        self.playing = False
        self.animation = None
        self.fitness_marker = None

        # Conectar eventos de teclado
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Dibujar gráfico de fitness (antes de update_display para inicializar marker)
        self._draw_fitness_plot()

        # Mostrar primer frame
        self._update_display()

        plt.show()

    def _update_display(self):
        """Actualiza la visualización con el frame actual."""
        gen, image, metrics = self._get_frame_data(self.current_frame)

        # Actualizar imagen principal
        self.ax_main.clear()
        self.ax_main.imshow(image)
        self.ax_main.set_title(f"Generación {gen}", fontsize=12)
        self.ax_main.axis("off")

        # Actualizar información
        self.ax_info.clear()
        self.ax_info.axis("off")
        info_text = self._format_metrics(gen, metrics)
        self.ax_info.text(
            0.1,
            0.95,
            info_text,
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        # Actualizar marcador en gráfico de fitness
        self._update_fitness_marker(gen)

        self.fig.canvas.draw_idle()

    def _draw_fitness_plot(self):
        """Dibuja el gráfico de evolución del fitness."""
        self.ax_plot.clear()

        if self.history:
            generations = [h.get("generation", i) for i, h in enumerate(self.history)]
            best_fitness = [h.get("best_fitness", 0) for h in self.history]
            avg_fitness = [h.get("avg_fitness", 0) for h in self.history]

            self.ax_plot.plot(
                generations, best_fitness, "b-", label="Mejor", linewidth=2
            )
            self.ax_plot.plot(
                generations, avg_fitness, "g--", label="Promedio", alpha=0.7
            )
            self.ax_plot.legend(loc="upper right")
        else:
            # Si no hay historial, mostrar los puntos de los frames
            frame_gens = [f[0] for f in self.frames]
            self.ax_plot.axvline(
                x=frame_gens[0], color="gray", linestyle="--", alpha=0.5
            )

        self.ax_plot.set_xlabel("Generación")
        self.ax_plot.set_ylabel("Fitness (MSE)")
        self.ax_plot.set_title("Evolución del Fitness")
        self.ax_plot.grid(True, alpha=0.3)

        # Guardar referencia para el marcador
        self.fitness_marker = None

    def _update_fitness_marker(self, gen: int):
        """Actualiza el marcador de posición en el gráfico."""
        if self.fitness_marker:
            self.fitness_marker.remove()

        self.fitness_marker = self.ax_plot.axvline(
            x=gen, color="red", linestyle="-", linewidth=2, alpha=0.7
        )

    def _on_slider_change(self, val):
        """Callback cuando cambia el slider."""
        self.current_frame = int(val)
        self._update_display()

    def _on_prev(self, event):
        """Callback para botón anterior."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.set_val(self.current_frame)

    def _on_next(self, event):
        """Callback para botón siguiente."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.slider.set_val(self.current_frame)

    def _on_play(self, event):
        """Callback para botón play/pause."""
        if self.playing:
            self.playing = False
            self.btn_play.label.set_text("▶ Play")
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.playing = True
            self.btn_play.label.set_text("⏸ Pause")
            self._animate()

    def _animate(self):
        """Inicia la animación automática."""

        def update(frame):
            if not self.playing:
                return
            if self.current_frame < len(self.frames) - 1:
                self.current_frame += 1
                self.slider.set_val(self.current_frame)
            else:
                self.playing = False
                self.btn_play.label.set_text("▶ Play")

        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=len(self.frames) - self.current_frame,
            interval=500,  # 500ms entre frames
            repeat=False,
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Callback para eventos de teclado."""
        if event.key == "left":
            self._on_prev(None)
        elif event.key == "right":
            self._on_next(None)
        elif event.key == " ":
            self._on_play(None)
        elif event.key == "home":
            self.current_frame = 0
            self.slider.set_val(0)
        elif event.key == "end":
            self.current_frame = len(self.frames) - 1
            self.slider.set_val(self.current_frame)

    def export_gif(self, output_path: str, fps: int = 2, duration_final: int = 3):
        """
        Exporta la evolución como GIF animado.

        Args:
            output_path: Ruta del archivo GIF.
            fps: Frames por segundo.
            duration_final: Segundos extra mostrando el resultado final.
        """
        print(f"Generando GIF con {len(self.frames)} frames...")

        images = []
        for gen, img_path in self.frames:
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        # Agregar frames extra del resultado final
        if images:
            for _ in range(fps * duration_final):
                images.append(images[-1].copy())

        # Guardar GIF
        duration_ms = int(1000 / fps)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )

        print(f"GIF guardado en: {output_path}")

    def export_video(self, output_path: str, fps: int = 5):
        """
        Exporta la evolución como video MP4.

        Args:
            output_path: Ruta del archivo de video.
            fps: Frames por segundo.
        """
        try:
            import cv2
        except ImportError:
            print("Error: opencv-python no está instalado.")
            print("Instálalo con: pip install opencv-python")
            return

        print(f"Generando video con {len(self.frames)} frames...")

        # Obtener dimensiones del primer frame
        first_img = Image.open(self.frames[0][1])
        width, height = first_img.size

        # Crear VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for gen, img_path in self.frames:
            img = Image.open(img_path).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(frame)

        # Agregar frames extra del final
        final_frame = cv2.cvtColor(
            np.array(Image.open(self.frames[-1][1])), cv2.COLOR_RGB2BGR
        )
        for _ in range(fps * 3):  # 3 segundos extra
            video.write(final_frame)

        video.release()
        print(f"Video guardado en: {output_path}")

    def create_summary_image(self, output_path: str, cols: int = 5):
        """
        Crea una imagen resumen mostrando la progresión.

        Args:
            output_path: Ruta de la imagen de salida.
            cols: Número de columnas en la grilla.
        """
        n_frames = len(self.frames)
        rows = (n_frames + cols - 1) // cols

        # Obtener tamaño de un frame
        sample_img = Image.open(self.frames[0][1])
        thumb_width, thumb_height = sample_img.size

        # Reducir tamaño para el resumen
        thumb_width = thumb_width // 2
        thumb_height = thumb_height // 2

        # Crear imagen de resumen
        padding = 10
        total_width = cols * thumb_width + (cols + 1) * padding
        total_height = (
            rows * thumb_height + (rows + 1) * padding + 30
        )  # +30 para títulos

        summary = Image.new("RGB", (total_width, total_height), "white")

        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(summary)

        for i, (gen, img_path) in enumerate(self.frames):
            row = i // cols
            col = i % cols

            x = padding + col * (thumb_width + padding)
            y = padding + row * (thumb_height + padding + 20)

            # Cargar y redimensionar
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

            # Pegar en el resumen
            summary.paste(img, (x, y + 15))

            # Agregar etiqueta de generación
            draw.text((x, y), f"Gen {gen}", fill="black")

        summary.save(output_path)
        print(f"Imagen resumen guardada en: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Visualizador de la evolución del algoritmo genético",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controles del visualizador interactivo:
  ← / →      Navegar entre generaciones
  Espacio    Play/Pause automático
  Home       Ir al inicio
  End        Ir al final

Ejemplos:
  python visualize.py output/mi_experimento
  python visualize.py output/mi_experimento --export-gif evolucion.gif
  python visualize.py output/mi_experimento --summary resumen.png
        """,
    )

    parser.add_argument(
        "output_dir", type=str, help="Directorio con los resultados del algoritmo"
    )

    parser.add_argument(
        "--export-gif", type=str, metavar="FILE", help="Exportar como GIF animado"
    )

    parser.add_argument(
        "--export-video", type=str, metavar="FILE", help="Exportar como video MP4"
    )

    parser.add_argument(
        "--summary",
        type=str,
        metavar="FILE",
        help="Crear imagen resumen con todos los pasos",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames por segundo para GIF/video (default: 2)",
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="No mostrar visualizador interactivo",
    )

    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()

    try:
        visualizer = EvolutionVisualizer(args.output_dir)
        print(f"Cargados {len(visualizer.frames)} frames de {args.output_dir}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Exportar si se solicita
    if args.export_gif:
        visualizer.export_gif(args.export_gif, fps=args.fps)

    if args.export_video:
        visualizer.export_video(args.export_video, fps=args.fps)

    if args.summary:
        visualizer.create_summary_image(args.summary)

    # Mostrar visualizador interactivo
    if not args.no_interactive:
        print("\nIniciando visualizador interactivo...")
        print("Controles: ← → (navegar), Espacio (play/pause), Home/End (inicio/fin)")
        visualizer.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
