"""Animador pygame para replays de Sokoban.

Sprites:
  - Caja:            ferno.png  (Fernet-Branca)
  - Objetivo:        vaso.png   (vaso con hielo)
  - Caja en goal:    vaso (80 %) + ferno encima
  - Pared:           rectángulo gris oscuro
  - Jugador:         joven (cabeza + cuerpo + piernas, dibujado)
  - HUD:             paso actual, algoritmo, resultado
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------
C_BG = (200, 180, 140)
C_WALL = (60, 60, 60)
C_WALL_HI = (90, 90, 90)
C_GOAL_RING = (218, 165, 32)  # aro dorado cuando no hay sprite
C_PLAYER_SKIN = (255, 200, 140)
C_PLAYER_SHIRT = (30, 100, 200)
C_PLAYER_PANTS = (40, 40, 100)
C_HUD_BG = (30, 30, 30)
C_HUD_TEXT = (255, 255, 255)
C_SUCCESS = (60, 200, 80)
C_FAILURE = (220, 60, 60)
C_ON_GOAL = (80, 220, 100)  # borde verde cuando ferno está en el objetivo

ASSETS_DIR = Path(__file__).parent / "assets"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
@dataclass
class ReplayData:
    algorithm: str
    board_path: str
    success: bool
    cost: int
    expanded_nodes: int
    time_elapsed: float
    rows: int
    cols: int
    walls: set[tuple[int, int]]
    goals: set[tuple[int, int]]
    moves: list[str]
    frames: list[dict]


def load_replay(path: str | Path) -> ReplayData:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["metadata"]
    board = data["board"]
    return ReplayData(
        algorithm=meta["algorithm"],
        board_path=meta["board_path"],
        success=meta["success"],
        cost=meta["cost"],
        expanded_nodes=meta["expanded_nodes"],
        time_elapsed=meta["time_elapsed"],
        rows=board["rows"],
        cols=board["cols"],
        walls={(r, c) for r, c in board["walls"]},
        goals={(r, c) for r, c in board["goals"]},
        moves=data["moves"],
        frames=data["frames"],
    )


# ---------------------------------------------------------------------------
# Sprites (imágenes escaladas)
# ---------------------------------------------------------------------------
@dataclass
class Sprites:
    box: pygame.Surface  # ferno normal
    box_on_goal: pygame.Surface  # vaso (pequeño) + ferno encima
    goal: pygame.Surface  # vaso solo


def _scale(img: pygame.Surface, size: int) -> pygame.Surface:
    return pygame.transform.smoothscale(img, (size, size))


def load_sprites(cell: int) -> Sprites | None:
    """Carga ferno.png y vaso.png desde assets/. Retorna None si faltan."""
    try:
        raw_ferno = pygame.image.load(ASSETS_DIR / "ferno.png").convert_alpha()
        raw_vaso = pygame.image.load(ASSETS_DIR / "vaso.png").convert_alpha()
    except (FileNotFoundError, pygame.error):
        return None

    box = _scale(raw_ferno, cell)
    goal = _scale(raw_vaso, cell)

    # Caja sobre objetivo: vaso al 75 % centrado + ferno encima
    box_on_goal = pygame.Surface((cell, cell), pygame.SRCALPHA)
    vaso_small = _scale(raw_vaso, int(cell * 0.75))
    offset = (cell - vaso_small.get_width()) // 2
    box_on_goal.blit(vaso_small, (offset, offset))
    box_on_goal.blit(box, (0, 0))
    # Borde verde para destacar que está en el objetivo
    pygame.draw.rect(box_on_goal, C_ON_GOAL, (0, 0, cell, cell), 3)

    return Sprites(box=box, box_on_goal=box_on_goal, goal=goal)


# ---------------------------------------------------------------------------
# Funciones de dibujo
# ---------------------------------------------------------------------------


def draw_wall(surf: pygame.Surface, row: int, col: int, cell: int) -> None:
    x, y = col * cell, row * cell
    pygame.draw.rect(surf, C_WALL, (x, y, cell, cell))
    pygame.draw.rect(surf, C_WALL_HI, (x + 2, y + 2, cell - 4, cell - 4), 2)


def draw_goal(
    surf: pygame.Surface,
    row: int,
    col: int,
    cell: int,
    sprites: Sprites | None,
) -> None:
    x, y = col * cell, row * cell
    if sprites:
        surf.blit(sprites.goal, (x, y))
    else:
        # Fallback: copa dibujada
        cx, cy = x + cell // 2, y + cell // 2
        r = cell // 5
        pygame.draw.circle(surf, (255, 215, 0), (cx, cy - r // 2), r)
        pygame.draw.circle(surf, C_GOAL_RING, (cx, cy - r // 2), r, 2)
        base_w = r + 4
        pygame.draw.rect(surf, C_GOAL_RING, (cx - base_w // 2, cy + r // 2, base_w, 3))
        pygame.draw.rect(surf, C_GOAL_RING, (cx - 2, cy - r // 2 + r, 4, r))


def draw_box(
    surf: pygame.Surface,
    row: int,
    col: int,
    cell: int,
    on_goal: bool,
    sprites: Sprites | None,
) -> None:
    x, y = col * cell, row * cell
    if sprites:
        img = sprites.box_on_goal if on_goal else sprites.box
        surf.blit(img, (x, y))
    else:
        # Fallback: botella dibujada
        pad = cell // 6
        bx, by = x + pad, y + pad // 2
        bw, bh = cell - pad * 2, cell - pad
        color = (10, 120, 40) if on_goal else (20, 80, 30)
        pygame.draw.rect(surf, color, (bx, by, bw, bh), border_radius=4)
        neck_w = bw // 2
        neck_h = bh // 5
        pygame.draw.rect(
            surf, color, (bx + bw // 2 - neck_w // 2, by - neck_h + 2, neck_w, neck_h)
        )
        lbl = 4
        pygame.draw.rect(
            surf,
            (240, 240, 240),
            (bx + lbl, by + bh // 3, bw - lbl * 2, bh // 3),
            border_radius=2,
        )


def draw_player(surf: pygame.Surface, row: int, col: int, cell: int) -> None:
    cx = col * cell + cell // 2
    cy = row * cell + cell // 2
    s = cell / 64

    head_r = int(10 * s)
    head_cy = cy - int(14 * s)
    pygame.draw.circle(surf, C_PLAYER_SKIN, (cx, head_cy), head_r)

    body_top = head_cy + head_r
    body_h = int(18 * s)
    body_w = int(16 * s)
    pygame.draw.rect(
        surf,
        C_PLAYER_SHIRT,
        (cx - body_w // 2, body_top, body_w, body_h),
        border_radius=3,
    )

    leg_top = body_top + body_h
    leg_h = int(14 * s)
    leg_w = int(6 * s)
    pygame.draw.rect(
        surf, C_PLAYER_PANTS, (cx - leg_w - 1, leg_top, leg_w, leg_h), border_radius=2
    )
    pygame.draw.rect(
        surf, C_PLAYER_PANTS, (cx + 1, leg_top, leg_w, leg_h), border_radius=2
    )


def draw_hud(
    surf: pygame.Surface,
    replay: ReplayData,
    frame_idx: int,
    hud_h: int,
    board_h: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    total_steps = len(replay.frames) - 1
    pygame.draw.rect(surf, C_HUD_BG, (0, board_h, surf.get_width(), hud_h))

    txt1 = (
        f"Algoritmo: {replay.algorithm.upper()}"
        f"   |   Tablero: {Path(replay.board_path).name}"
    )
    surf.blit(font.render(txt1, True, C_HUD_TEXT), (10, board_h + 6))

    move_txt = replay.moves[frame_idx - 1] if frame_idx > 0 else "INICIO"
    txt2 = f"Paso {frame_idx}/{total_steps}   Movimiento: {move_txt}"
    surf.blit(small_font.render(txt2, True, C_HUD_TEXT), (10, board_h + 28))

    txt3 = (
        f"Costo: {replay.cost}   "
        f"Nodos expandidos: {replay.expanded_nodes}   "
        f"Tiempo: {replay.time_elapsed:.4f}s"
    )
    surf.blit(small_font.render(txt3, True, C_HUD_TEXT), (10, board_h + 46))

    result_color = C_SUCCESS if replay.success else C_FAILURE
    result_txt = "EXITO" if replay.success else "FRACASO"
    surf.blit(
        small_font.render(result_txt, True, result_color),
        (surf.get_width() - 90, board_h + 28),
    )


def draw_frame(
    surf: pygame.Surface,
    replay: ReplayData,
    frame_idx: int,
    cell: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    hud_h: int,
    sprites: Sprites | None,
) -> None:
    board_h = replay.rows * cell
    frame = replay.frames[frame_idx]
    player_pos = tuple(frame["player"])
    box_set = {tuple(b) for b in frame["boxes"]}

    surf.fill(C_BG)

    for r in range(replay.rows):
        for c in range(replay.cols):
            pos = (r, c)
            if pos in replay.walls:
                draw_wall(surf, r, c, cell)
            elif pos in replay.goals:
                on_goal = pos in box_set
                draw_goal(surf, r, c, cell, sprites)
                if on_goal:
                    draw_box(surf, r, c, cell, on_goal=True, sprites=sprites)
            elif pos in box_set:
                draw_box(surf, r, c, cell, on_goal=False, sprites=sprites)

    draw_player(surf, int(player_pos[0]), int(player_pos[1]), cell)
    draw_hud(surf, replay, frame_idx, hud_h, board_h, font, small_font)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


def run_animation(
    replay_path: str | Path,
    cell_size: int = 64,
    fps: float = 2.0,
) -> None:
    replay = load_replay(replay_path)

    pygame.init()
    pygame.display.set_caption(
        f"Sokoban — {replay.algorithm.upper()} — {Path(replay_path).name}"
    )

    hud_h = 72
    width = replay.cols * cell_size
    height = replay.rows * cell_size + hud_h

    screen = pygame.display.set_mode((width, height))

    # load_sprites DEBE ir después de set_mode: convert_alpha() necesita el display
    sprites = load_sprites(cell_size)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)

    frame_idx = 0
    total_frames = len(replay.frames)
    ms_per_frame = int(1000 / fps)
    elapsed = 0
    finished = False

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not finished:
            elapsed += dt
            if elapsed >= ms_per_frame:
                elapsed = 0
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                else:
                    finished = True

        draw_frame(
            screen, replay, frame_idx, cell_size, font, small_font, hud_h, sprites
        )
        pygame.display.flip()

    pygame.quit()
