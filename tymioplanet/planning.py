from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from heapq import heappop, heappush

import numpy as np
from scipy import ndimage

THYMIO_RADIUS = 0.08  # m. Thymio II ~ 11 x 11 cm, cercle circonscrit autour du centre des roues
HELD_KARP_MAX = 12    # au-dela, Held-Karp (2^n * n^2) devient trop lent en Python


# ==============================================================================================
# 1. SERIE 5 : heuristique + grid_search (Dijkstra / A*)
# ==============================================================================================
def heuristic(a, b):
    """Serie 5 : distance de Manhattan pour une grille 4-connexe.
    AJOUT : en 8-connexe on prend la distance euclidienne (toujours admissible, ne surestime pas)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Serie 5 : 4 voisins (haut, bas, gauche, droite), chaque pas a une longueur 1
NEIGHBORS_4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
# AJOUT : 8 voisins (+ diagonales de longueur sqrt(2)), utilise pour le Thymio
NEIGHBORS_8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
               (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def grid_search(map_grid, S, G=None, algo="Dijkstra", neighbors=NEIGHBORS_8):
    """Correction de la serie 5, etendue. Renvoie (path, explored, operation_count, g_costs, came_from).

    - Cout d'un pas : serie 5 = 1 + map_grid[voisin] ;
      ici = longueur du pas (1 ou sqrt(2)) * (1 + map_grid[voisin])   (AJOUT : diagonales).
    - AJOUT : G = None -> on ne s'arrete pas a un but : Dijkstra explore toute la carte et
      g_costs contient la distance de S a CHAQUE case atteignable (path = None).
    - neighbors=NEIGHBORS_4 redonne EXACTEMENT la correction de la serie 5.
    """
    ## initialize the variables (comme dans la serie 5)
    came_from = {}       # to reconstruct path
    g_costs = {S: 0.0}   # cost from start to the cell
    explored = set()     # to keep track of explored cells
    operation_count = 0  # to count the number of operations
    if algo == "Dijkstra":
        open_set = [(0.0, S)]                        # priority queue (cost, position)
    if algo == "A*":
        open_set = [(heuristic(S, G), 0.0, S)]       # priority queue (f_cost, g_cost, position)

    current_pos = None
    while open_set:  # G goal is unmarked
        if algo == "Dijkstra":
            current_g_cost, current_pos = heappop(open_set)
        if algo == "A*":
            current_f_cost, current_g_cost, current_pos = heappop(open_set)

        # AJOUT : une case peut etre dans la file plusieurs fois (ancienne valeur plus chere) ;
        # on ignore les anciennes copies -> chaque case n'est developpee qu'une fois
        if current_pos in explored:
            continue
        explored.add(current_pos)

        if current_pos == G:  # if G goal is marked
            break

        for di, dj, step in neighbors:  # for each neighbor of marked cells
            neighbor = (current_pos[0] + di, current_pos[1] + dj)
            # Check if neighbor is within bounds
            if not (0 <= neighbor[0] < map_grid.shape[0] and 0 <= neighbor[1] < map_grid.shape[1]):
                continue
            # Check if neighbor is not an obstacle
            if map_grid[neighbor] == -1:
                continue
            # AJOUT : pas de diagonale qui coupe le coin d'un obstacle
            if di and dj and (map_grid[current_pos[0] + di, current_pos[1]] == -1
                              or map_grid[current_pos[0], current_pos[1] + dj] == -1):
                continue
            tentative_g_cost = current_g_cost + step * (1 + map_grid[neighbor])
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current_pos
                operation_count += 1
                if algo == "Dijkstra":
                    heappush(open_set, (tentative_g_cost, neighbor))
                if algo == "A*":
                    f_cost = tentative_g_cost + heuristic(neighbor, G)
                    heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # Reconstruct path (serie 5)
    path = reconstruct_path(came_from, S, G) if (G is not None and current_pos == G) else None
    return path, explored, operation_count, g_costs, came_from


def reconstruct_path(came_from, S, G):
    """Serie 5 : on part de G et on remonte came_from jusqu'a S."""
    if G != S and G not in came_from:
        return None
    current_pos, path = G, []
    while current_pos != S:
        path.append(current_pos)
        current_pos = came_from[current_pos]
    path.append(S)
    path.reverse()
    return path


# ==============================================================================================
# 2. AJOUT : configuration space + carte ponderee (le "sable" = la proximite des obstacles)
# ==============================================================================================
def configuration_space(occ, res, robot_radius, margin):
    """Renvoie (blocked, clearance).
    clearance[i, j] = distance (m) du centre de la case a l'obstacle le plus proche (bords = murs).
    blocked = clearance < r + marge : le CENTRE du Thymio ne peut pas aller dans cette case."""
    padded = np.pad(np.asarray(occ, bool), 1, constant_values=True)
    clearance = ndimage.distance_transform_edt(~padded)[1:-1, 1:-1] * res
    return clearance < (robot_radius + margin), clearance


def build_map_grid(blocked, clearance, w_clear=2.0, d_safe=0.06, robot_radius=THYMIO_RADIUS):
    """Carte au format de la serie 5 : -1 = obstacle, sinon le poids de la case.
    poids = w * pen,  pen = clip(1 - (clearance - r) / d_safe, 0, 1)
      pen = 0 si on a plus de d_safe de marge, pen = 1 si on frole l'obstacle.
    Un pas vers cette case coute alors longueur * (1 + poids) (cf. grid_search)."""
    pen = np.clip(1.0 - (clearance - robot_radius) / d_safe, 0.0, 1.0)
    map_grid = w_clear * pen
    map_grid[blocked] = -1
    return map_grid


# ==============================================================================================
# 3. AJOUT : ordre de visite (TSP chemin ouvert : start -> tous les targets -> end)
# ==============================================================================================
def held_karp(D, start, end, targets):
    """TSP exact par programmation dynamique. Renvoie (ordre, cout).
    C({k}, k) = D[start, k]
    C(S, k)   = min_{m in S \\ {k}} C(S \\ {k}, m) + D[m, k]
    cout*     = min_k C(tous, k) + D[k, end]          -> O(n^2 2^n)"""
    n = len(targets)
    if n == 0:
        return [], float(D[start, end])
    t = np.asarray(targets)
    Dt = D[np.ix_(t, t)]
    full = (1 << n) - 1
    C = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=int)
    for k in range(n):
        C[1 << k, k] = D[start, t[k]]
    for S in range(1, full + 1):  # S croissant -> les sous-ensembles sont deja calcules
        for k in range(n):
            if not (S >> k) & 1 or not np.isfinite(C[S, k]):
                continue
            for j in range(n):  # on prolonge le chemin (S, fini en k) vers j hors de S
                if (S >> j) & 1:
                    continue
                S2, c = S | (1 << j), C[S, k] + Dt[k, j]
                if c < C[S2, j]:
                    C[S2, j], parent[S2, j] = c, k
    final = C[full] + D[t, end]
    k = int(np.argmin(final))
    best = float(final[k])
    if not np.isfinite(best):
        return None, math.inf
    order, S = [], full
    while k >= 0:
        order.append(int(t[k]))
        S, k = S & ~(1 << k), parent[S, k]
    return order[::-1], best


def tour_cost(D, start, end, order):
    seq = [start, *order, end]
    return float(sum(D[a, b] for a, b in zip(seq[:-1], seq[1:])))


def nn_two_opt(D, start, end, targets):
    """Heuristique pour beaucoup de ballons : plus proche voisin puis 2-opt."""
    left, order, cur = set(targets), [], start
    while left:
        nxt = min(left, key=lambda j: D[cur, j])
        order.append(nxt)
        left.remove(nxt)
        cur = nxt
    improved = True
    while improved:
        improved = False
        for a, b in itertools.combinations(range(len(order)), 2):
            cand = order[:a] + order[a:b + 1][::-1] + order[b + 1:]
            if tour_cost(D, start, end, cand) < tour_cost(D, start, end, order) - 1e-12:
                order, improved = cand, True
    return order, tour_cost(D, start, end, order)


# ==============================================================================================
# 4. AJOUT : le planificateur de mission (ballons, replanning)
# ==============================================================================================
@dataclass
class Balloon:
    name: str
    xy: tuple[float, float]
    color: str  # "red" (asteroide, cible) / "green" (planete, a eviter)


@dataclass
class Plan:
    order: list[str]        # asteroides dans l'ordre de visite (sans depart ni arrivee)
    cost: float             # cout total (m "ponderes")
    path: np.ndarray        # chemin complet (K x 2) en metres, case par case
    waypoints: np.ndarray   # chemin simplifie (N x 2, lignes droites) pour le controleur
    legs: list[np.ndarray] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # 1 label par waypoint :
    #   "depart", "passage" (virage pour contourner un obstacle), "A2" (asteroide a eclater), "arrivee"

    def waypoint_list(self, decimals=3):
        """Ce qu'on donne au motion control : [(x, y), ...] en metres, du depart a l'arrivee."""
        return [(round(float(x), decimals), round(float(y), decimals)) for x, y in self.waypoints]


class MissionPlanner:
    """Regle : les ballons de couleur `avoid_colors` (planetes vertes) sont des obstacles
    (disque de rayon `balloon_radius`), les autres (asteroides rouges) sont les cibles."""

    def __init__(self, occ, res, start_xy, goal_xy=None, balloons=(), robot_radius=THYMIO_RADIUS,
                 margin=0.02, w_clear=2.0, d_safe=0.06, hysteresis=0.05,
                 avoid_colors=("green",), balloon_radius=0.05, waypoint_mode="ligne_de_vue"):
        # start = point de depart du Thymio, goal = point d'arrivee (si None : retour au depart)
        self.res, self.start = res, tuple(start_xy)
        self.goal = tuple(start_xy if goal_xy is None else goal_xy)
        self.robot_radius, self.margin = robot_radius, margin
        self.w_clear, self.d_safe, self.hysteresis = w_clear, d_safe, hysteresis
        self.avoid_colors, self.balloon_radius = tuple(avoid_colors), balloon_radius
        # "ligne_de_vue" : on coupe les coins des qu'une ligne droite libre existe (moins de points)
        # "coins"        : un point a CHAQUE changement de direction du chemin Dijkstra
        self.waypoint_mode = waypoint_mode
        self.balloons: dict[str, Balloon] = {b.name: b for b in balloons}
        self.current: Plan | None = None
        self.set_map(occ)

    @property
    def earth(self):  # ancien nom de l'arrivee (garde pour thymio_live.py)
        return self.goal

    @earth.setter
    def earth(self, xy):
        self.goal = tuple(xy)

    def is_hazard(self, name):
        return self.balloons[name].color in self.avoid_colors

    def targets(self):
        return [n for n in self.balloons if not self.is_hazard(n)]

    # --- evenements qui declenchent un replanning -----------------------------------------
    def set_map(self, occ):
        """Nouvelle carte (camera) : obstacles sans les ballons."""
        self.occ_base = np.asarray(occ, bool)
        self._rebuild()

    def _disc(self, xy, radius):
        h, w = self.occ_base.shape
        yy, xx = (np.mgrid[0:h, 0:w] + 0.5) * self.res
        return (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= radius ** 2

    def _rebuild(self):
        """Carte = obstacles camera + planetes, puis C-space et map_grid (format serie 5)."""
        occ = self.occ_base.copy()
        for n in self.balloons:
            if self.is_hazard(n):
                occ |= self._disc(self.balloons[n].xy, self.balloon_radius)
        self.occ = occ
        self.blocked, self.clearance = configuration_space(
            occ, self.res, self.robot_radius, self.margin)
        self.map_grid = build_map_grid(self.blocked, self.clearance, self.w_clear,
                                       self.d_safe, self.robot_radius)

    def add_obstacle_disc(self, xy, radius):
        self.set_map(self.occ_base | self._disc(xy, radius))

    def add_balloon(self, name, xy, color):
        self.balloons[name] = Balloon(name, tuple(xy), color)
        if color in self.avoid_colors:
            self._rebuild()

    def popped(self, name):
        """Asteroide eclate (ou ballon retire de l'arene)."""
        b = self.balloons.pop(name, None)
        if b is not None and b.color in self.avoid_colors:
            self._rebuild()

    # --- outils grille ----------------------------------------------------------------------
    def _cell(self, xy):
        h, w = self.occ.shape
        return (int(np.clip(xy[1] / self.res, 0, h - 1)), int(np.clip(xy[0] / self.res, 0, w - 1)))

    def _snap(self, xy):
        """Case libre la plus proche (un ballon colle a un mur tombe dans la zone grossie)."""
        i, j = self._cell(xy)
        if self.blocked[i, j]:
            _, (ii, jj) = ndimage.distance_transform_edt(self.blocked, return_indices=True)
            i, j = int(ii[i, j]), int(jj[i, j])
        return (i, j)

    def _xy(self, cell):
        return ((cell[1] + 0.5) * self.res, (cell[0] + 0.5) * self.res)

    def _los_free(self, a, b):
        """Ligne droite a-b sans case bloquee."""
        n = int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * self.res))) + 1
        xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
        h, w = self.occ.shape
        i = np.clip((ys / self.res).astype(int), 0, h - 1)
        j = np.clip((xs / self.res).astype(int), 0, w - 1)
        return not self.blocked[i, j].any()

    def corners(self, path):
        """Waypoints mode "coins" : debut, fin et chaque case ou la direction du chemin change."""
        path = np.asarray(path)
        if len(path) < 3:
            return path
        d = np.round(np.diff(path, axis=0) / self.res).astype(int)   # pas (di, dj) entre cases
        turn = np.any(d[1:] != d[:-1], axis=1)                       # la direction change ici
        keep = np.r_[True, turn, True]
        return path[keep]

    def simplify(self, path):
        """Waypoints : on ne garde que les points ou il faut tourner (line-of-sight)."""
        if self.waypoint_mode == "coins":
            return self.corners(path)
        if len(path) < 3:
            return np.asarray(path)
        out, k = [path[0]], 0
        while k < len(path) - 1:
            m = len(path) - 1
            while m > k + 1 and not self._los_free(path[k], path[m]):
                m -= 1
            out.append(path[m])
            k = m
        return np.asarray(out)

    # --- planification ----------------------------------------------------------------------
    def _costs(self, pose_xy):
        """Points : 0 = robot (pose actuelle), 1..n = asteroides, n+1 = arrivee.
        Un grid_search (Dijkstra complet, G = None) par point -> matrice D en metres."""
        names = self.targets()
        pts = [pose_xy] + [self.balloons[n].xy for n in names] + [self.goal]
        cells = [self._snap(p) for p in pts]
        D = np.full((len(cells), len(cells)), np.inf)
        came = []
        for a, S in enumerate(cells):
            _, _, _, g_costs, came_from = grid_search(self.map_grid, S, G=None, algo="Dijkstra")
            came.append(came_from)
            for b, cell in enumerate(cells):
                if cell in g_costs:
                    D[a, b] = g_costs[cell] * self.res  # cases -> metres
        return names, cells, D, came

    def _solve(self, pose_xy):
        names, cells, D, came = self._costs(pose_xy)
        n = len(names)
        start, end = 0, n + 1
        if not np.isfinite(D[start, end]):
            raise RuntimeError("L'arrivee n'est pas atteignable depuis la pose actuelle.")
        reach = [k for k in range(1, n + 1) if np.isfinite(D[start, k])]
        unreachable = [names[k - 1] for k in range(1, n + 1) if k not in reach]
        solver = held_karp if len(reach) <= HELD_KARP_MAX else nn_two_opt
        order, cost = solver(D, start, end, reach)
        return names, cells, D, came, order, cost, unreachable

    def plan(self, pose_xy=None):
        """Plan optimal depuis pose_xy (defaut : le depart), sans tenir compte du plan courant."""
        pose_xy = self.start if pose_xy is None else pose_xy
        names, cells, D, came, order, cost, unreachable = self._solve(pose_xy)
        return self._build_plan(names, cells, D, came, order, cost, unreachable)

    def _build_plan(self, names, cells, D, came, order, cost, unreachable):
        seq = [0, *order, len(names) + 1]
        legs = []
        for a, b in zip(seq[:-1], seq[1:]):
            path_cells = reconstruct_path(came[a], cells[a], cells[b])  # came_from de la serie 5
            legs.append(np.array([self._xy(c) for c in path_cells]))
        path = np.vstack([legs[0]] + [lg[1:] for lg in legs[1:]])
        wps = [self.simplify(lg) for lg in legs]
        waypoints = np.vstack([wps[0]] + [w[1:] for w in wps[1:]])
        # label de chaque waypoint : la fin de chaque troncon est un asteroide (ou l'arrivee)
        labels = ["depart"]
        for b, w in zip(seq[1:], wps):
            lab = names[b - 1] if b <= len(names) else "arrivee"
            if len(w) < 2:   # troncon de longueur nulle (deja sur place) : meme point que le precedent
                labels[-1] = lab
            else:
                labels += ["passage"] * (len(w) - 2) + [lab]
        return Plan([names[k - 1] for k in order], cost, path, waypoints, legs, unreachable, labels)

    def replan(self, pose_xy=None):
        """Recalcule l'ordre optimal depuis la pose actuelle (defaut : le depart). Garde l'ancien
        ordre si le nouveau n'est pas au moins `hysteresis` (5 %) meilleur. Renvoie (plan, changed)."""
        pose_xy = self.start if pose_xy is None else pose_xy
        names, cells, D, came, order, cost, unreachable = self._solve(pose_xy)
        start, end = 0, len(names) + 1
        old = self.current
        if old is not None:
            keep = [x for x in old.order if x in self.balloons and x not in unreachable]
            if set(keep) == {names[k - 1] for k in order}:
                k_order = [names.index(x) + 1 for x in keep]
                k_cost = tour_cost(D, start, end, k_order)
                if np.isfinite(k_cost) and cost > (1 - self.hysteresis) * k_cost:
                    order, cost = k_order, k_cost
        new = self._build_plan(names, cells, D, came, order, cost, unreachable)
        changed = old is None or [x for x in old.order if x in self.balloons] != new.order
        self.current = new
        return new, changed



# ==============================================================================================
# 5. SORTIE pour les camarades (motion control) : la liste de waypoints
# ==============================================================================================
def save_waypoints(plan, filename="waypoints.csv"):
    """Ecrit les waypoints dans un CSV :  index, x_m, y_m, label   (ouvrable avec Excel / pandas)."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("index,x_m,y_m,label\n")
        for k, ((x, y), lab) in enumerate(zip(plan.waypoint_list(), plan.labels)):
            f.write(f"{k},{x},{y},{lab}\n")
    return filename


def compute_waypoints(occ, res, start_xy, goal_xy, asteroids, planets=(), **kw):
    """Fonction "tout-en-un" pour l'integration.
    occ       : grille booleenne (True = obstacle), occ[i, j] avec i -> y, j -> x
    res       : taille d'une case en m (ex. 0.01)
    start_xy  : (x, y) du depart en m          goal_xy : (x, y) de l'arrivee en m
    asteroids : liste de (x, y) a visiter      planets : liste de (x, y) a eviter
    Renvoie (waypoints, plan) avec waypoints = [(x, y), ...] en m, du depart a l'arrivee."""
    balloons = [Balloon(f"A{k + 1}", tuple(p), "red") for k, p in enumerate(asteroids)]
    balloons += [Balloon(f"P{k + 1}", tuple(p), "green") for k, p in enumerate(planets)]
    mp = MissionPlanner(occ, res, start_xy, goal_xy, balloons, **kw)
    plan, _ = mp.replan()
    return plan.waypoint_list(), plan


# ==============================================================================================
# Verification sur la carte de la serie 5 (python mission_planner.py)
# ==============================================================================================
if __name__ == "__main__":
    Map = np.zeros((11, 9))
    Map[5, 3] = 2  # Weighted cell
    Map[0, :] = -1; Map[10, :] = -1; Map[:, 0] = -1; Map[:, 8] = -1  # noqa: E702
    Map[8, 1] = -1; Map[9, 1] = -1; Map[9, 2] = -1; Map[4:6, 4:8] = -1  # noqa: E702
    S, G = (8, 5), (2, 6)
    for nb_name, nb in (("4 voisins (= serie 5)", NEIGHBORS_4), ("8 voisins (projet)", NEIGHBORS_8)):
        for algo in ("Dijkstra", "A*"):
            path, explored, ops, g, _ = grid_search(Map, S, G, algo, nb)
            print(f"{nb_name:22s} {algo:8s} cout = {g[G]:5.2f}  cases explorees = {len(explored):2d}")
    print("-> la correction de la serie 5 donne un cout de 13 avec 4 voisins.")
