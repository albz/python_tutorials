"""
Simulazione 2D di palline in un contenitore circolare, pronta per Google Colab.

Regole richieste:
- Contenitore: disco di raggio R.
- Inizio con 2 palline posizionate random nell'area superiore (y > 2/3 * R).
- Gravità verso il basso.
- Urti elastici senza attrito:
  * pallina-pallina: conservazione di quantità di moto ed energia cinetica (e=1)
  * pallina-parete circolare: riflessione speculare perfettamente elastica
- Le palline hanno raggio finito (non punti materiali), quindi collisioni sui bordi.
- Ogni volta che due palline si toccano, viene generata una nuova pallina nel punto di contatto.
- Stop automatico quando il disco è quasi pieno (frazione area occupata >= soglia).

Uso in Colab:
1) Copia questo file in una cella (o eseguilo direttamente con `!python nome_file.py`).
2) In Colab usa **prima** `run_animation_jshtml()` (animazione HTML/JS inline).
3) Se vuoi frame-by-frame, usa `run_animation_colab_live()`.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# =============================
# Parametri fisici/simulazione
# =============================
R_CONTAINER = 1.0          # raggio del contenitore circolare
BALL_RADIUS = 0.045        # raggio delle palline
BALL_MASS = 1.0            # massa (uguale per tutte)
GRAVITY = 2.4              # accelerazione gravitazionale (unità arbitrarie)
DT = 0.004                 # passo temporale
SUBSTEPS = 4               # sub-step per robustezza numerica

INITIAL_BALLS = 2
MAX_BALLS = 500
FILL_THRESHOLD = 0.84      # stop quando area occupata/area contenitore supera soglia

# Per evitare spawning multiplo continuo nello stesso contatto
PAIR_COOLDOWN_STEPS = 15

# Velocità iniziali casuali (moderate)
INIT_SPEED_MIN = 0.05
INIT_SPEED_MAX = 0.25

RNG_SEED = 42


@dataclass
class Ball:
    pos: np.ndarray   # shape (2,) -> [x, y]
    vel: np.ndarray   # shape (2,) -> [vx, vy]
    radius: float
    mass: float


class CircleCollisionSimulation:
    def __init__(self, rng_seed: int = RNG_SEED):
        self.rng = np.random.default_rng(rng_seed)
        self.balls: List[Ball] = []
        self.step_count = 0
        self.stopped = False

        # cooldown collisioni tra coppie (i,j)
        self.pair_cooldown: Dict[Tuple[int, int], int] = {}

        self._init_balls()

    def _init_balls(self):
        """Inizializza 2 palline nella zona alta del contenitore: y > 2/3 R."""
        y_min = (2.0 / 3.0) * R_CONTAINER

        attempts_limit = 20_000
        for _ in range(INITIAL_BALLS):
            placed = False
            for _attempt in range(attempts_limit):
                r = (R_CONTAINER - BALL_RADIUS) * np.sqrt(self.rng.uniform(0.0, 1.0))
                theta = self.rng.uniform(0.0, 2.0 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                if y <= y_min:
                    continue

                candidate_pos = np.array([x, y], dtype=float)
                if self._overlaps_existing(candidate_pos, BALL_RADIUS):
                    continue

                speed = self.rng.uniform(INIT_SPEED_MIN, INIT_SPEED_MAX)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)

                self.balls.append(
                    Ball(
                        pos=candidate_pos,
                        vel=np.array([vx, vy], dtype=float),
                        radius=BALL_RADIUS,
                        mass=BALL_MASS,
                    )
                )
                placed = True
                break

            if not placed:
                raise RuntimeError("Impossibile posizionare le palline iniziali senza sovrapposizioni.")

        # Forziamo una dinamica iniziale che favorisca il primo contatto.
        if len(self.balls) >= 2:
            b0, b1 = self.balls[0], self.balls[1]
            direction = b1.pos - b0.pos
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                u = direction / norm
                v0 = self.rng.uniform(0.15, 0.3)
                v1 = self.rng.uniform(0.15, 0.3)
                b0.vel = u * v0
                b1.vel = -u * v1

    def _overlaps_existing(self, pos: np.ndarray, radius: float) -> bool:
        for b in self.balls:
            d = np.linalg.norm(pos - b.pos)
            if d < (radius + b.radius):
                return True
        return False

    def fill_fraction(self) -> float:
        occupied = len(self.balls) * math.pi * BALL_RADIUS**2
        total = math.pi * R_CONTAINER**2
        return occupied / total

    def _resolve_wall_collision(self, ball: Ball):
        """
        Collisione pallina-parete (contenitore circolare):
        - se il centro è oltre R-r, lo riportiamo sul bordo interno
        - riflettiamo la velocità rispetto alla normale locale
        """
        dist = np.linalg.norm(ball.pos)
        max_center_dist = R_CONTAINER - ball.radius

        if dist > max_center_dist:
            if dist == 0.0:
                normal = np.array([0.0, 1.0])
            else:
                normal = ball.pos / dist

            # correzione geometrica (nessuna compenetrazione)
            ball.pos = normal * max_center_dist

            # riflessione elastica: v' = v - 2*(v·n) n
            vn = np.dot(ball.vel, normal)
            if vn > 0:
                ball.vel = ball.vel - 2.0 * vn * normal

    def _pair_key(self, i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _can_spawn_from_pair(self, i: int, j: int) -> bool:
        key = self._pair_key(i, j)
        return self.pair_cooldown.get(key, -10**9) <= (self.step_count - PAIR_COOLDOWN_STEPS)

    def _mark_pair_spawned(self, i: int, j: int):
        self.pair_cooldown[self._pair_key(i, j)] = self.step_count

    def _resolve_ball_collision(self, i: int, j: int):
        """
        Urto elastico tra due dischi (2D):
        - separazione geometrica in caso di overlap
        - impulso lungo la normale di collisione
        - conservazione momento + energia cinetica (e=1)
        """
        bi = self.balls[i]
        bj = self.balls[j]

        delta = bj.pos - bi.pos
        dist = np.linalg.norm(delta)
        min_dist = bi.radius + bj.radius

        if dist == 0.0:
            # direzione casuale per uscire dalla singolarità
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            n = np.array([np.cos(angle), np.sin(angle)])
            dist = 1e-12
        else:
            n = delta / dist

        if dist > min_dist:
            return

        # 1) Correzione di compenetrazione (split in proporzione alle masse)
        overlap = min_dist - dist
        total_mass = bi.mass + bj.mass
        bi_correction = overlap * (bj.mass / total_mass)
        bj_correction = overlap * (bi.mass / total_mass)

        bi.pos -= n * bi_correction
        bj.pos += n * bj_correction

        # 2) Spawn nuova pallina nel punto di contatto se possibile
        if (
            len(self.balls) < MAX_BALLS
            and self._can_spawn_from_pair(i, j)
            and self.fill_fraction() < FILL_THRESHOLD
        ):
            contact_point = bi.pos + n * bi.radius

            # La nuova pallina eredita velocità del centro di massa locale
            v_new = (bi.mass * bi.vel + bj.mass * bj.vel) / (bi.mass + bj.mass)

            # Richiesta utente: nascita esattamente nel punto di contatto.
            self.balls.append(
                Ball(
                    pos=contact_point.copy(),
                    vel=v_new.copy(),
                    radius=BALL_RADIUS,
                    mass=BALL_MASS,
                )
            )
            self._mark_pair_spawned(i, j)

        # 3) Impulso elastico lungo normale
        rel_vel = bj.vel - bi.vel
        rel_normal_speed = np.dot(rel_vel, n)

        # Se stanno già separandosi, niente impulso
        if rel_normal_speed >= 0:
            return

        e = 1.0  # coefficiente restituzione elastico perfetto
        j_impulse = -(1.0 + e) * rel_normal_speed / (1.0 / bi.mass + 1.0 / bj.mass)

        bi.vel -= (j_impulse / bi.mass) * n
        bj.vel += (j_impulse / bj.mass) * n

    def _can_place_new_ball(self, candidate: Ball) -> bool:
        # dentro al contenitore
        if np.linalg.norm(candidate.pos) + candidate.radius > R_CONTAINER:
            return False

        # no overlap con palline esistenti (tolleranza numerica)
        tol = 1e-6
        for b in self.balls:
            d = np.linalg.norm(candidate.pos - b.pos)
            if d < (candidate.radius + b.radius - tol):
                return False
        return True


    def step(self):
        if self.stopped:
            return

        dt = DT / SUBSTEPS

        for _ in range(SUBSTEPS):
            # integrazione semplice semi-implicita
            for b in self.balls:
                b.vel[1] -= GRAVITY * dt
                b.pos += b.vel * dt

            # collisioni con parete
            for b in self.balls:
                self._resolve_wall_collision(b)

            # collisioni tra palline
            n_balls_at_start = len(self.balls)
            for i in range(n_balls_at_start):
                for j in range(i + 1, n_balls_at_start):
                    self._resolve_ball_collision(i, j)

        self.step_count += 1

        # stop quando quasi pieno
        if self.fill_fraction() >= FILL_THRESHOLD or len(self.balls) >= MAX_BALLS:
            self.stopped = True


def run_animation(total_frames: int = 4_000):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    sim = CircleCollisionSimulation()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    lim = R_CONTAINER + 0.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Collisioni elastiche in contenitore circolare")

    # Disegno contenitore
    container_patch = Circle((0, 0), R_CONTAINER, fill=False, linewidth=2.0)
    ax.add_patch(container_patch)

    # Collezione patch palline (dinamica)
    ball_patches = []

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    def sync_patches_with_balls():
        while len(ball_patches) < len(sim.balls):
            c = Circle((0, 0), BALL_RADIUS, alpha=0.9)
            ax.add_patch(c)
            ball_patches.append(c)

        for k, patch in enumerate(ball_patches):
            if k < len(sim.balls):
                patch.set_visible(True)
                patch.center = tuple(sim.balls[k].pos)
            else:
                patch.set_visible(False)

    def init():
        sync_patches_with_balls()
        info_text.set_text("Init")
        return [container_patch, info_text, *ball_patches]

    def update(_frame):
        if not sim.stopped:
            sim.step()

        sync_patches_with_balls()

        info = (
            f"palline: {len(sim.balls)}\n"
            f"riempimento: {sim.fill_fraction():.1%}\n"
            f"step: {sim.step_count}\n"
            f"stopped: {sim.stopped}"
        )
        info_text.set_text(info)

        return [container_patch, info_text, *ball_patches]

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=16,
        blit=True,
        repeat=False,
    )

    return fig, ani


def run_animation_jshtml(total_frames: int = 4_000, fps: int = 60):
    """
    Modalità consigliata per Google Colab.
    Restituisce e mostra HTML/JS animato nel notebook (non immagine statica).
    """
    from IPython.display import HTML, display

    fig, ani = run_animation(total_frames=total_frames)
    html = HTML(ani.to_jshtml(fps=fps))
    display(html)
    return html


def run_animation_colab_live(max_steps: int = 8_000, draw_every: int = 2):
    """
    Aggiornamento progressivo frame-by-frame per Colab/Jupyter.
    Se il browser/renderer del notebook blocca gli update live, usa run_animation_jshtml().
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    from matplotlib.patches import Circle

    sim = CircleCollisionSimulation()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    lim = R_CONTAINER + 0.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Collisioni elastiche in contenitore circolare (live)")

    container_patch = Circle((0, 0), R_CONTAINER, fill=False, linewidth=2.0)
    ax.add_patch(container_patch)

    ball_patches = []
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    def sync_patches_with_balls():
        while len(ball_patches) < len(sim.balls):
            c = Circle((0, 0), BALL_RADIUS, alpha=0.9)
            ax.add_patch(c)
            ball_patches.append(c)
        for k, patch in enumerate(ball_patches):
            if k < len(sim.balls):
                patch.set_visible(True)
                patch.center = tuple(sim.balls[k].pos)
            else:
                patch.set_visible(False)

    for tick in range(max_steps):
        if not sim.stopped:
            sim.step()

        if tick % draw_every == 0 or sim.stopped:
            sync_patches_with_balls()
            info_text.set_text(
                f"palline: {len(sim.balls)}\n"
                f"riempimento: {sim.fill_fraction():.1%}\n"
                f"step: {sim.step_count}\n"
                f"stopped: {sim.stopped}"
            )
            clear_output(wait=True)
            display(fig)
            plt.pause(0.001)

        if sim.stopped:
            break

    return sim


if __name__ == "__main__":
    # Se eseguito come script (`!python ...`) in Colab, l'HTML inline non viene reso.
    # Mostriamo comunque una preview matplotlib classica e stampiamo istruzioni corrette.
    import matplotlib.pyplot as plt

    fig, _ani = run_animation(total_frames=1200)
    plt.show()
    print("Per Colab in tempo reale: esegui in una cella Python `run_animation_jshtml()`.")
