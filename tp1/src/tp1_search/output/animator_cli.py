"""Entry point para tp1-animate.

Uso:
    uv run tp1-animate <replay.json> [--speed FPS] [--cell-size PX] [--save-gif OUT.gif]
"""

import argparse
import sys
from pathlib import Path

from tp1_search.output.pygame_anim import export_gif, run_animation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animador pygame para replays de Sokoban — TP1 SIA"
    )
    parser.add_argument(
        "replay",
        help="Ruta al archivo JSON de replay generado por tp1-search --save-replay",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        metavar="FPS",
        help="Velocidad de la animación en frames por segundo (default: 2)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=64,
        metavar="PX",
        help="Tamaño de cada celda en píxeles (default: 64)",
    )
    parser.add_argument(
        "--save-gif",
        metavar="OUT.gif",
        help="Exporta la animación a un GIF y termina sin abrir la ventana",
    )
    args = parser.parse_args()

    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"Error: archivo de replay no encontrado: {replay_path}", file=sys.stderr)
        sys.exit(1)

    if args.save_gif:
        output_path = export_gif(
            replay_path=replay_path,
            output_path=args.save_gif,
            cell_size=args.cell_size,
            fps=args.speed,
        )
        print(f"GIF guardado en: {output_path}")
        return

    run_animation(
        replay_path=replay_path,
        cell_size=args.cell_size,
        fps=args.speed,
    )


if __name__ == "__main__":
    main()
