"""Animador pygame para replays de Sokoban.

Sprites:
  - Caja:            ferno.png  (Fernet-Branca)
  - Objetivo:        vaso.png   (vaso con hielo)
  - Caja en goal:    llenito.png
  - Pared:           rectángulo gris oscuro
  - Jugador:         sprites direccionales up/down/left/right
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
C_PLAYER_HAIR = (190, 40, 30)
C_PLAYER_SHIRT = (30, 100, 200)
C_PLAYER_PANTS = (40, 40, 100)
C_HUD_BG = (30, 30, 30)
C_HUD_TEXT = (255, 255, 255)
C_SUCCESS = (60, 200, 80)
C_FAILURE = (220, 60, 60)
C_ON_GOAL = (80, 220, 100)  # borde verde cuando ferno está en el objetivo
C_WOOD_TOP = (116, 78, 50)
C_WOOD_BOTTOM = (70, 45, 28)
C_WOOD_SEAM = (58, 36, 22)
C_WOOD_GRAIN = (145, 104, 72)
HUD_HEIGHT = 118

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
    box_on_goal: pygame.Surface  # llenito
    goal: pygame.Surface  # vaso solo
    player: dict[str, pygame.Surface] | None = None


def _scale(img: pygame.Surface, size: int) -> pygame.Surface:
    return pygame.transform.smoothscale(img, (size, size))


def _fit_sprite_to_cell(
    img: pygame.Surface, cell: int, padding: float = 0.08
) -> pygame.Surface:
    bounds = img.get_bounding_rect(min_alpha=1)
    if bounds.width > 0 and bounds.height > 0:
        img = img.subsurface(bounds).copy()

    inner = max(1, int(round(cell * (1.0 - padding * 2))))
    scale = min(inner / img.get_width(), inner / img.get_height())
    target_w = max(1, int(round(img.get_width() * scale)))
    target_h = max(1, int(round(img.get_height() * scale)))
    scaled = pygame.transform.smoothscale(img, (target_w, target_h))

    canvas = pygame.Surface((cell, cell), pygame.SRCALPHA)
    offset_x = (cell - target_w) // 2
    offset_y = (cell - target_h) // 2
    canvas.blit(scaled, (offset_x, offset_y))
    return canvas


def _load_optional_player_sprites(cell: int) -> dict[str, pygame.Surface] | None:
    names = {
        "UP": "up.png",
        "DOWN": "down.png",
        "LEFT": "left.png",
        "RIGHT": "right.png",
    }
    try:
        raw = {
            direction: pygame.image.load(ASSETS_DIR / filename).convert_alpha()
            for direction, filename in names.items()
        }
    except (FileNotFoundError, pygame.error):
        return None

    return {
        direction: _fit_sprite_to_cell(surface, cell)
        for direction, surface in raw.items()
    }


def load_sprites(cell: int) -> Sprites | None:
    """Carga sprites de cajas/goals y, si existen, sprites direccionales del jugador."""
    try:
        raw_ferno = pygame.image.load(ASSETS_DIR / "ferno.png").convert_alpha()
        raw_vaso = pygame.image.load(ASSETS_DIR / "vaso.png").convert_alpha()
        raw_llenito = pygame.image.load(ASSETS_DIR / "llenito.png").convert_alpha()
    except (FileNotFoundError, pygame.error):
        return None

    box = _fit_sprite_to_cell(raw_ferno, cell)
    goal = _fit_sprite_to_cell(raw_vaso, cell)
    box_on_goal = _fit_sprite_to_cell(raw_llenito, cell)
    player = _load_optional_player_sprites(cell)

    return Sprites(box=box, box_on_goal=box_on_goal, goal=goal, player=player)


def _lerp_color(
    start: tuple[int, int, int], end: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return (
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    )


def _generate_wood_background(width: int, height: int) -> pygame.Surface:
    bg = pygame.Surface((width, height)).convert()

    for y in range(height):
        t = y / max(1, height - 1)
        pygame.draw.line(
            bg, _lerp_color(C_WOOD_TOP, C_WOOD_BOTTOM, t), (0, y), (width, y)
        )

    plank_h = max(24, height // 7)
    for plank_idx, top in enumerate(range(0, height, plank_h)):
        plank_rect = pygame.Rect(0, top, width, min(plank_h, height - top))
        tint = 8 if plank_idx % 2 == 0 else -8
        plank_color = tuple(
            max(0, min(255, c + tint))
            for c in _lerp_color(C_WOOD_TOP, C_WOOD_BOTTOM, 0.35)
        )
        pygame.draw.rect(bg, plank_color, plank_rect, 0)
        pygame.draw.line(bg, C_WOOD_SEAM, (0, top), (width, top), 2)

        grain_step = max(18, width // 12)
        for x in range((plank_idx % 3) * 7, width + grain_step, grain_step):
            pygame.draw.line(
                bg,
                C_WOOD_GRAIN,
                (x, top + 4),
                (
                    min(width - 1, x + plank_h // 3),
                    min(height - 1, top + plank_rect.height - 4),
                ),
                1,
            )

    vignette = pygame.Surface((width, height), pygame.SRCALPHA)
    border = max(18, min(width, height) // 16)
    pygame.draw.rect(vignette, (0, 0, 0, 0), (0, 0, width, height))
    pygame.draw.rect(vignette, (0, 0, 0, 48), (0, 0, width, border))
    pygame.draw.rect(vignette, (0, 0, 0, 48), (0, height - border, width, border))
    pygame.draw.rect(vignette, (0, 0, 0, 42), (0, 0, border, height))
    pygame.draw.rect(vignette, (0, 0, 0, 42), (width - border, 0, border, height))
    bg.blit(vignette, (0, 0))

    return bg


def load_background(width: int, height: int, cell: int) -> pygame.Surface:
    """Carga floor.png y lo repite por celda; si falta, usa madera generada."""
    try:
        raw_floor = pygame.image.load(ASSETS_DIR / "floor.png").convert()
    except (FileNotFoundError, pygame.error):
        return _generate_wood_background(width, height)

    floor_tile = _scale(raw_floor, cell)
    bg = pygame.Surface((width, height)).convert()

    for y in range(0, height, cell):
        for x in range(0, width, cell):
            bg.blit(floor_tile, (x, y))

    return bg


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


def draw_player(
    surf: pygame.Surface,
    row: int,
    col: int,
    cell: int,
    sprites: Sprites | None,
    direction: str,
) -> None:
    x, y = col * cell, row * cell
    if sprites and sprites.player:
        img = sprites.player.get(direction)
        if img is not None:
            surf.blit(img, (x, y))
            return

    cx = col * cell + cell // 2
    cy = row * cell + cell // 2
    s = cell / 64

    head_r = int(10 * s)
    head_cy = cy - int(14 * s)
    pygame.draw.circle(surf, C_PLAYER_SKIN, (cx, head_cy), head_r)
    pygame.draw.circle(surf, C_PLAYER_HAIR, (cx, head_cy - int(3 * s)), int(7 * s))

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
    current_fps: float | None = None,
) -> None:
    def _fit_text(text: str, text_font: pygame.font.Font, max_width: int) -> str:
        if text_font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        trimmed = text
        while trimmed and text_font.size(trimmed + ellipsis)[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + ellipsis) if trimmed else ellipsis

    total_steps = len(replay.frames) - 1
    width = surf.get_width()
    left_pad = 10
    right_pad = 10
    pygame.draw.rect(surf, C_HUD_BG, (0, board_h, width, hud_h))

    algo_txt = f"Algoritmo: {replay.algorithm.upper()}"
    surf.blit(font.render(algo_txt, True, C_HUD_TEXT), (left_pad, board_h + 6))

    board_txt = f"Tablero: {Path(replay.board_path).name}"
    board_txt = _fit_text(board_txt, small_font, width - left_pad - right_pad)
    surf.blit(small_font.render(board_txt, True, C_HUD_TEXT), (left_pad, board_h + 24))

    move_txt = replay.moves[frame_idx - 1] if frame_idx > 0 else "INICIO"
    move_line = f"Paso {frame_idx}/{total_steps} | Mov: {move_txt}"
    surf.blit(small_font.render(move_line, True, C_HUD_TEXT), (left_pad, board_h + 42))

    result_color = C_SUCCESS if replay.success else C_FAILURE
    result_txt = "EXITO" if replay.success else "FRACASO"
    result_surface = small_font.render(result_txt, True, result_color)
    surf.blit(
        result_surface, (width - right_pad - result_surface.get_width(), board_h + 42)
    )

    cost_line = f"Costo: {replay.cost}"
    surf.blit(small_font.render(cost_line, True, C_HUD_TEXT), (left_pad, board_h + 60))

    stats_line = (
        f"Expandidos: {replay.expanded_nodes} | Tiempo: {replay.time_elapsed:.4f}s"
    )
    stats_line = _fit_text(stats_line, small_font, width - left_pad - right_pad)
    surf.blit(small_font.render(stats_line, True, C_HUD_TEXT), (left_pad, board_h + 78))

    if current_fps is not None:
        speed_txt = f"Velocidad: {current_fps:.1f} FPS | UP/DOWN"
        speed_txt = _fit_text(speed_txt, small_font, width - left_pad - right_pad)
        surf.blit(
            small_font.render(speed_txt, True, C_HUD_TEXT), (left_pad, board_h + 96)
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
    current_fps: float | None = None,
    board_bg: pygame.Surface | None = None,
) -> None:
    board_h = replay.rows * cell
    frame = replay.frames[frame_idx]
    player_pos = tuple(frame["player"])
    box_set = {tuple(b) for b in frame["boxes"]}
    if frame_idx > 0:
        player_dir = replay.moves[frame_idx - 1]
    elif replay.moves:
        player_dir = replay.moves[0]
    else:
        player_dir = "DOWN"
    if player_dir not in {"UP", "DOWN", "LEFT", "RIGHT"}:
        player_dir = "DOWN"

    if board_bg is not None:
        surf.blit(board_bg, (0, 0))
    else:
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

    draw_player(
        surf,
        int(player_pos[0]),
        int(player_pos[1]),
        cell,
        sprites,
        player_dir,
    )
    draw_hud(surf, replay, frame_idx, hud_h, board_h, font, small_font, current_fps)


def _build_render_assets(
    replay: ReplayData,
    cell_size: int,
) -> tuple[
    int, int, int, pygame.Surface, Sprites | None, pygame.font.Font, pygame.font.Font
]:
    hud_h = HUD_HEIGHT
    width = replay.cols * cell_size
    board_h = replay.rows * cell_size
    height = board_h + hud_h
    board_bg = load_background(width, board_h, cell_size)
    sprites = load_sprites(cell_size)
    font = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)
    return width, height, hud_h, board_bg, sprites, font, small_font


def export_gif(
    replay_path: str | Path,
    output_path: str | Path,
    cell_size: int = 64,
    fps: float = 2.0,
) -> Path:
    """Renderiza un replay completo y lo exporta como GIF animado."""
    from PIL import Image

    replay = load_replay(replay_path)
    output_path = Path(output_path)
    current_fps = max(_FPS_MIN, min(_FPS_MAX, fps))
    frame_duration_ms = max(1, int(round(1000 / current_fps)))

    pygame.init()
    try:
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
        width, height, hud_h, board_bg, sprites, font, small_font = (
            _build_render_assets(replay, cell_size)
        )
        frame_surface = pygame.Surface((width, height)).convert()

        pil_frames = []
        for frame_idx in range(len(replay.frames)):
            draw_frame(
                frame_surface,
                replay,
                frame_idx,
                cell_size,
                font,
                small_font,
                hud_h,
                sprites,
                current_fps,
                board_bg,
            )
            raw = pygame.image.tobytes(frame_surface, "RGB")
            frame_image = Image.frombytes("RGB", (width, height), raw)
            frame_image.info = {}
            pil_frames.append(frame_image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        first, *rest = pil_frames
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        pygame.quit()

    return output_path


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


_FPS_MIN = 0.5
_FPS_MAX = 30.0
_FPS_STEP = 0.5


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

    hud_h = HUD_HEIGHT
    width = replay.cols * cell_size
    height = replay.rows * cell_size + hud_h
    screen = pygame.display.set_mode((width, height))
    width, height, hud_h, board_bg, sprites, font, small_font = _build_render_assets(
        replay, cell_size
    )
    clock = pygame.time.Clock()

    frame_idx = 0
    total_frames = len(replay.frames)
    current_fps = max(_FPS_MIN, min(_FPS_MAX, fps))
    ms_per_frame = int(1000 / current_fps)
    elapsed = 0
    paused = False
    finished = False

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_UP:
                    current_fps = min(_FPS_MAX, current_fps + _FPS_STEP)
                    ms_per_frame = int(1000 / current_fps)
                elif event.key == pygame.K_DOWN:
                    current_fps = max(_FPS_MIN, current_fps - _FPS_STEP)
                    ms_per_frame = int(1000 / current_fps)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    frame_idx = 0
                    elapsed = 0
                    finished = False
                    paused = False
                elif event.key == pygame.K_RIGHT and paused:
                    if frame_idx < total_frames - 1:
                        frame_idx += 1
                elif event.key == pygame.K_LEFT and paused:
                    if frame_idx > 0:
                        frame_idx -= 1

        if not finished and not paused:
            elapsed += dt
            if elapsed >= ms_per_frame:
                elapsed = 0
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                else:
                    finished = True

        draw_frame(
            screen,
            replay,
            frame_idx,
            cell_size,
            font,
            small_font,
            hud_h,
            sprites,
            current_fps,
            board_bg,
        )
        pygame.display.flip()

    pygame.quit()
