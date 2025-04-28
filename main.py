import random
import time
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ——————————————————————————————————————————————————————————————————
#                   REPRODUCIBILITY & RANDOM SEEDING
# ——————————————————————————————————————————————————————————————————
SEED = 1                          # generates reproducible results
random.seed(SEED)
np.random.seed(SEED)
print(f"Using fixed SEED={SEED}")

# ——————————————————————————————————————————————————————————————————
#                      HYPERPARAMETERS & TOGGLES
# ——————————————————————————————————————————————————————————————————
NUM_CUSTOMERS = 31                # Number of customer locations (depot is node 0)
X_BOUNDS = (0, 50)                # Min/max X coordinate for random node generation
Y_BOUNDS = (0, 50)                # Min/max Y coordinate for random node generation

NUM_ANTS = 150                    # Colony size: number of ants per iteration
ALPHA, BETA = 1.5, 2.5            # Pheromone vs. heuristic weighting exponents
RHO, Q = 0.5, 120.0               # Evaporation rate and pheromone deposit factor
INITIAL_PHEROMONE = 1.0           # Starting pheromone level on all edges

USE_CANDIDATE_LIST = True         # Toggle to enable nearest‐neighbor candidate lists
CANDIDATE_LIST_SIZE = 10          # Size of neighbor list per node (if enabled)

ITERATIONS = 99                  # Total ACO iterations to perform
DYNAMIC_EVERY = 20                # Inject a new node into the graph every N iterations
BENCHMARK_RELATIVE = True         # Track relative performance change after each injection
ENABLE_REALTIME_PLOT = False       # Toggle live updating of route plot during iterations

# plotting controls
PLOT_UPDATE_FREQ = 50             # Only refresh the real‐time plot every N iterations

# Local‐search (3‑opt) cadence
THREE_OPT_CADENCE = 5             # Apply 3‑opt to the global best every N iterations

# Selective 2‑opt refining parameters
TOP_K_FRACTION = 0.1              # Fraction of best ants to refine each iteration
MIN_TOP_K = 10                    # Minimum number of ants to apply 2‑opt to
BUDGETED_2OPT_TRIALS = 20         # Number of random 2‑opt trials per polished tour

# Figure sizes for Matplotlib
FIGSIZE_ROUTE = (7, 7)            # Width, height for the real‐time route plot
FIGSIZE_CONV = (8, 4)             # Width, height for the convergence‐curve plot
FIGSIZE_RELATIVE = (8, 4)         # Width, height for the injection‐relative bar plot

# ——————————————————————————————————————————————————————————————————
#                        NUMBA CONFIGURATION
# ——————————————————————————————————————————————————————————————————
try:
    import numba

    USE_NUMBA = True
    print("Numba detected. Using JIT compilation.")
except ImportError:
    numba = None  # type: ignore

    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    class DummyNumba:
        pass

    numba = DummyNumba()  # type: ignore
    numba.njit = njit  # type: ignore
    USE_NUMBA = False
    print("Numba not found. Running pure Python.")

# ——————————————————————————————————————————————————————————————————
#                        NUMBA‑ACCELERATED HELPERS
# ——————————————————————————————————————————————————————————————————
@numba.njit(cache=True, fastmath=True)
def numba_weighted_choice(cands: np.ndarray, probs: np.ndarray) -> int:
    """
    Perform roulette‐wheel selection over `cands` using unnormalized `probs`.
    Returns the chosen candidate index, or -1 if no valid choice.
    """
    s = probs.sum()                               # total weight = sum of pheromone×heuristic products
    if s <= 0:
        if len(cands) > 0:                        # no usable weight → fallback to uniform random pick
            idx0 = np.random.randint(0, len(cands))
            return int(cands[np.int64(idx0)])
        else:
            return -1                             # no candidates available
    cum = np.cumsum(probs)                        # build cumulative distribution array
    r = np.random.rand() * s                      # sample a random threshold in [0, total weight)
    idx = np.searchsorted(cum, r, side='right')   # find first bin where cumulative ≥ r
    return int(cands[min(idx, len(cands) - 1)])


@numba.njit(cache=True, fastmath=True)
def numba_tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    tot = 0.0
    N = dist.shape[0]
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        if 0 <= u < N and 0 <= v < N:
            tot += dist[u, v]
        else:
            return -1.0
    return tot


@numba.njit(cache=True, fastmath=True)
def numba_update_pheromones_deposit(
        tours_list: numba.typed.List,
        dist: np.ndarray,
        pher: np.ndarray,
        Q: float,
        N: int
):
    for t in tours_list:
        L = numba_tour_length(t, dist)
        if L <= 0 or np.isnan(L) or np.isinf(L):
            continue
        delta = Q / L
        for j in range(len(t) - 1):
            a, b = t[j], t[j + 1]
            if 0 <= a < N and 0 <= b < N:
                pher[a, b] += delta
                pher[b, a] += delta


@numba.njit(cache=True, fastmath=True)
def numba_two_opt(tour_np, dist):
    best = tour_np.copy()
    best_len = numba_tour_length(best, dist)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_t = np.empty_like(best)
                new_t[:i] = best[:i]
                for k in range(j, i - 1, -1):
                    new_t[i + j - k] = best[k]
                new_t[j + 1:] = best[j + 1:]
                L = numba_tour_length(new_t, dist)
                if L + 1e-9 < best_len:
                    best[:] = new_t
                    best_len = L
                    improved = True
    return best, best_len


@numba.njit(cache=True)
def numba_three_opt_segment(tour_np, dist, i, j, k, n):
    best_tour = tour_np.copy()
    best_len = numba_tour_length(tour_np, dist)
    variants = []
    # variant 1
    v1 = np.empty(n, dtype=np.int64)
    v1[:i] = tour_np[:i]
    for idx in range(j - 1, i - 1, -1):
        v1[i + j - 1 - idx] = tour_np[idx]
    v1[j:k] = tour_np[j:k]
    v1[k:] = tour_np[k:]
    variants.append(v1)
    # variant 2
    v2 = np.empty(n, dtype=np.int64)
    v2[:j] = tour_np[:j]
    for idx in range(k - 1, j - 1, -1):
        v2[j + k - 1 - idx] = tour_np[idx]
    v2[k:] = tour_np[k:]
    variants.append(v2)
    # variant 3
    v3 = np.empty(n, dtype=np.int64)
    v3[:i] = tour_np[:i]
    for idx in range(j - 1, i - 1, -1):
        v3[i + j - 1 - idx] = tour_np[idx]
    for idx in range(k - 1, j - 1, -1):
        v3[j + k - 1 - idx] = tour_np[idx]
    v3[k:] = tour_np[k:]
    variants.append(v3)
    # variant 4
    v4 = np.empty(n, dtype=np.int64)
    v4[:i] = tour_np[:i]
    v4[i:i + k - j] = tour_np[j:k]
    v4[i + k - j:i + k - j + j - i] = tour_np[i:j]
    v4[k:] = tour_np[k:]
    variants.append(v4)

    for v in variants:
        L = numba_tour_length(v, dist)
        if L + 1e-9 < best_len:
            best_tour = v.copy()
            best_len = L
    return best_tour, best_len


@numba.njit(cache=True, fastmath=True)
def numba_three_opt(tour, dist):
    best = tour.copy()
    best_len = numba_tour_length(best, dist)
    n = len(best)
    max_segments = min(100, n * (n - 1) * (n - 2) // 6)
    count = 0
    for _ in range(max_segments):
        i = np.random.randint(1, n - 4)
        j = np.random.randint(i + 2, n - 2)
        k = np.random.randint(j + 2, n)
        new_tour, new_len = numba_three_opt_segment(best, dist, i, j, k, n)
        if new_len < best_len:
            best = new_tour
            best_len = new_len
        count += 1
        if count >= max_segments:
            break
    return best, best_len


# ——————————————————————————————————————————————————————————————————
#                           PLOTTING HELPERS
# ——————————————————————————————————————————————————————————————————
@lru_cache(maxsize=1)
def setup_realtime_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=FIGSIZE_ROUTE)
    ax.set_title("ACO‑DVRP Real‑time Route")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(0.1)
    return fig, ax


def update_realtime_plot(
        ax: plt.Axes,
        coords: np.ndarray,
        tour: List[int],
        iteration: int,
        best_len: float
) -> bool:
    if not plt.fignum_exists(ax.figure.number):
        print("Plot closed; stopping updates.")
        return False
    ax.clear()
    xs, ys = coords[:, 0], coords[:, 1]
    if coords.shape[0] > 1:
        ax.scatter(xs[1:], ys[1:], c="red", label="Customers", s=25, zorder=3)
    ax.scatter(xs[0], ys[0], c="blue", label="Depot", s=100, zorder=3)
    if tour and len(tour) > 1:
        valid = [i for i in tour if 0 <= i < coords.shape[0]]
        valid_arr = np.array(valid, dtype=np.int64)
        ax.plot(coords[valid_arr, 0], coords[valid_arr, 1], "-o",
                c="lightblue", markersize=4, linewidth=1.5, zorder=2)
    ax.set_title(f"Iter {iteration} | Best {best_len:.2f}")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axis('equal')
    ax.figure.canvas.draw_idle()
    plt.pause(0.01)
    return True


def plot_convergence(hist: List[float]):
    plt.figure(figsize=FIGSIZE_CONV)
    plt.plot(hist, marker='.', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Best Length")
    plt.title("Convergence")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_convergence_and_relative(hist: List[float], rel: List[float], dynamic_every: int):
    plt.figure(figsize=FIGSIZE_CONV)
    plt.plot(hist, marker='.', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Best Length")
    plt.title("Convergence")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if rel:
        plt.figure(figsize=FIGSIZE_RELATIVE)
        iters = (np.arange(len(rel)) + 1) * dynamic_every
        plt.bar(iters, rel, width=dynamic_every * 0.8, color="orange", edgecolor="k")
        plt.xlabel("Injection Iter")
        plt.ylabel("Relative Increase")
        plt.title("Relative Increase per Injection")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()


def plot_relative(rel: List[float], dynamic_every: int):
    if not rel:
        print("No relative data.")
        return
    plt.figure(figsize=FIGSIZE_RELATIVE)
    iters = (np.arange(len(rel)) + 1) * dynamic_every
    plt.bar(iters, rel, width=dynamic_every * 0.8, color="orange", edgecolor="k")
    plt.xlabel("Injection Iter")
    plt.ylabel("Relative Increase")
    plt.title("Relative Increase per Injection")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_route(
        coords: np.ndarray,
        tour: List[int],
        title: str = "Final Best Route",
        runtime: Optional[float] = None
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_ROUTE)
    xs, ys = coords[:, 0], coords[:, 1]
    if coords.shape[0] > 1:
        ax.scatter(xs[1:], ys[1:], c="red", label="Customers", s=25, zorder=3)
    ax.scatter(xs[0], ys[0], c="blue", label="Depot", s=100, zorder=3)
    if tour:
        valid_arr = np.array([i for i in tour if 0 <= i < coords.shape[0]], dtype=np.int64)
        ax.plot(coords[valid_arr, 0], coords[valid_arr, 1], "-o",
                c="lightblue", markersize=4, linewidth=1.5, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    if runtime is not None:
        ax.text(0.02, 0.98, f"Total runtime: {runtime:.2f}s",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.tight_layout()


# ——————————————————————————————————————————————————————————————————
#                       DATA & DYNAMIC UPDATE
# ——————————————————————————————————————————————————————————————————
def generate_random_nodes(n, xb, yb):
    depot = np.array([[0., 0.]])
    customer_coords = np.random.uniform(
        low=[xb[0], yb[0]],
        high=[xb[1], yb[1]],
        size=(n, 2)
    )
    return np.vstack([depot, customer_coords])


@lru_cache(maxsize=1)
def compute_distance_matrix(coords_tuple):
    coords = np.array(coords_tuple)
    dx = coords[:, np.newaxis, 0] - coords[np.newaxis, :, 0]
    dy = coords[:, np.newaxis, 1] - coords[np.newaxis, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def dynamic_add_node(aco: "ACO_DVRP"):
    aco._injection_count += 1
    rnd = SEED + aco._injection_count
    random.seed(rnd)
    np.random.seed(rnd)
    x0, x1 = X_BOUNDS
    y0, y1 = Y_BOUNDS
    new_pt = np.array([[random.uniform(x0, x1), random.uniform(y0, y1)]])
    aco.coords = np.vstack([aco.coords, new_pt])
    coords_ = tuple(map(tuple, aco.coords))
    aco.dist = compute_distance_matrix(coords_)


# ——————————————————————————————————————————————————————————————————
#                      LOCAL SEARCH FUNCTIONS
# ——————————————————————————————————————————————————————————————————
def two_opt(tour, dist):
    """
    Perform 2‑opt local search on a given tour.
    - If Numba is available, delegate to a JIT‑compiled routine for best performance.
    - Otherwise, repeatedly reverse any segment (i,j) that shortens the tour until no improvement.
    """
    if USE_NUMBA:
        tour_np = np.array(tour, dtype=np.int64)                    # Convert to numpy array for the JIT function
        result, _ = numba_two_opt(tour_np, dist)                    # numba_two_opt returns (improved_tour, improved_length)
        return result.tolist()
    else:
        best, best_len = tour, tour_length(tour, dist)              # Pure‑Python fallback
        n = len(best)
        improved = True                                             # Loop until no further segment reversal yields an improvement
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_t = best[:i] + best[i:j + 1][::-1] + best[j + 1:]        # Reverse the subtour between i and j
                    L = tour_length(new_t, dist)
                    if L + 1e-9 < best_len:                         # Accept the first better tour found (first‑improvement)
                        best, best_len = new_t, L
                        improved = True
        return best


def two_opt_budgeted(tour, dist, trials=BUDGETED_2OPT_TRIALS):
    best = tour
    best_len = tour_length(best, dist)
    n = len(best)
    for _ in range(trials):
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, n - 2)
        new_t = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
        L = tour_length(new_t, dist)
        if L + 1e-9 < best_len:
            best, best_len = new_t, L
    return best


def three_opt(tour, dist):
    """
    Perform 3‑opt local search on a given tour.
    - If Numba is available, delegate to a JIT‑compiled routine.
    - Otherwise, sample up to 100 random (i,j,k) cuts and test multiple reconnection variants.
    - Accept any variant that improves tour length.
    """
    if USE_NUMBA:
        tour_np = np.array(tour, dtype=np.int64)                # Convert to numpy array and call the JIT‑compiled 3‑opt
        result, _ = numba_three_opt(tour_np, dist)
        return result.tolist()
    else:
        best, best_len = tour, tour_length(tour, dist)          # Pure‑Python fallback implementation
        n = len(best)
        max_segments = min(100, n * (n - 1) * (n - 2) // 6)     # Limit the number of evaluated segment triplets
        for _ in range(max_segments):
            i = random.randint(1, n - 4)                     # Randomly pick three cut indices i < j < k
            j = random.randint(i + 2, n - 2)
            k = random.randint(j + 2, n)
            A = best[:i]
            B = best[i:j]
            C = best[j:k]
            D = best[k:]
            variants = [                                        # Build a handful of useful reconnection variants
                A + B[::-1] + C + D,         # reverse segment B
                A + B + C[::-1] + D,         # reverse segment C
                A + B[::-1] + C[::-1] + D,   # reverse both B and C
                A + C + B + D,               # swap segments B and C
                A + C[::-1] + B + D,         # reverse C then swap
                A + C + B[::-1] + D,         # reverse B then swap
                A + C[::-1] + B[::-1] + D    # reverse both then swap
            ]
            for new_t in variants:           # Test each candidate, accept improvement immediately
                if len(new_t) != n:
                    continue
                L = tour_length(new_t, dist)
                if L + 1e-9 < best_len:
                    best, best_len = new_t, L
        return best


def tour_length(t, dist):
    if isinstance(t, list) and len(t) > 1:
        arr = np.array(t, dtype=np.int64)
        return float(dist[arr[:-1], arr[1:]].sum())
    return float('inf')


# ——————————————————————————————————————————————————————————————————
#                              ACO CLASS
# ——————————————————————————————————————————————————————————————————
class ACO_DVRP:
    def __init__(self, dist_matrix, num_ants, alpha, beta, rho, Q, initial_pheromone):
        if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("dist_matrix must be square")
        if np.any(dist_matrix < 0):
            raise ValueError("Distances cannot be negative")
        self.dist = dist_matrix.copy()
        self.N = self.dist.shape[0]
        self.pher = np.full((self.N, self.N), initial_pheromone, dtype=np.float64)
        np.fill_diagonal(self.pher, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.dist > 0, 1.0 / self.dist, 0.0)
        self.num_ants, self.alpha, self.beta, self.rho, self.Q = (
            num_ants, alpha, beta, rho, Q
        )
        self._injection_count = 0
        self.neighbors_list = []
        if USE_CANDIDATE_LIST:
            self._build_candidate_list(CANDIDATE_LIST_SIZE)
        self.coords: Optional[np.ndarray] = None
        self.ant_tours = [[] for _ in range(num_ants)]
        if USE_NUMBA:
            print("Pre‑compiling Numba…")
            t0 = time.time()
            try:
                dummy = np.array([0, 0], dtype=np.int64)
                dm2 = self.dist[:2, :2] if self.N > 1 else np.array([[0.]])
                ph2 = self.pher[:2, :2] if self.N > 1 else np.array([[0.]])
                et2 = self.eta[:2, :2] if self.N > 1 else np.array([[0.]])
                vm = np.zeros(self.N, dtype=np.bool_)
                nbrs = [np.array([1], dtype=np.int64)] * self.N
                nbrs = numba.typed.List(nbrs)
                _ = self._build_tour_numba(0, vm, self.N, ph2, et2,
                                           self.alpha, self.beta,
                                           True, nbrs)
                _ = numba_tour_length(dummy, dm2)
                lst = numba.typed.List([dummy])
                _ = numba_update_pheromones_deposit(lst, dm2, ph2,
                                                    self.Q, dm2.shape[0])
                if self.N > 1:
                    tour_test = np.array([0, 1, 0], dtype=np.int64)
                    _, _ = numba_two_opt(tour_test, dm2)
                    _, _ = numba_three_opt(tour_test, dm2)
                print(f"Numba compile done in {(time.time() - t0):.2f}s")
            except Exception as e:
                print(f"Numba compile warning: {e}")

    def _build_candidate_list(self, k):
        k = min(k, self.N - 1)
        if k <= 0:
            self.neighbors_list = []
            return
        self.neighbors_list = numba.typed.List()
        order = np.argsort(self.dist, axis=1)
        for i in range(self.N):
            self.neighbors_list.append(order[i, 1:k + 1].astype(np.int64))

    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _build_tour_numba(start, vis, N, pher, eta, alpha, beta,
                          use_list, nbrs):
        tour = np.empty(N + 1, dtype=np.int64)
        tour[0] = start
        vis[start] = True
        curr, idx = start, 1
        for _ in range(N - 1):
            if use_list and curr < len(nbrs):
                cand = nbrs[curr]
                mask = ~vis[cand]
                cand = cand[mask]
                if int(cand.size) == 0:
                    cand = np.arange(N)[~vis]
            else:
                cand = np.arange(N)[~vis]
            if len(cand) == 0:
                break
            ph_pow = pher[curr, cand] ** alpha
            eta_pow = eta[curr, cand] ** beta
            pr = ph_pow * eta_pow
            s = pr.sum()
            if s <= 0 or not np.isfinite(s):
                nxt = cand[np.random.randint(0, len(cand))]
            else:
                nxt = numba_weighted_choice(cand, pr)
                if nxt == -1:
                    break
            tour[idx], vis[nxt], curr = nxt, True, nxt
            idx += 1
        if idx == N:
            tour[N] = start
            return tour
        out = np.empty(idx + 1, dtype=np.int64)
        out[:idx], out[idx] = tour[:idx], start
        return out

    def _build_tour(self) -> List[int]:
        if USE_NUMBA and self.neighbors_list is not None:
            vis = np.zeros(self.N, dtype=np.bool_)
            arr = self._build_tour_numba(0, vis, self.N, self.pher,
                                         self.eta, self.alpha, self.beta,
                                         True, self.neighbors_list)
            return arr.tolist()
        vis = np.zeros(self.N, dtype=bool)
        vis[0], tour = True, [0]
        curr = 0
        for _ in range(self.N - 1):
            if USE_CANDIDATE_LIST and self.neighbors_list and curr < len(self.neighbors_list):
                cand = self.neighbors_list[curr]
                mask = ~vis[cand]
                cand = cand[mask]
                if int(cand.size) == 0:
                    cand = np.where(~vis)[0]
            else:
                cand = np.where(~vis)[0]
            if int(cand.size) == 0:
                break
            ph_pow = self.pher[curr, cand] ** self.alpha
            eta_pow = self.eta[curr, cand] ** self.beta
            pr = ph_pow * eta_pow
            tot = pr.sum()
            if tot <= 0 or not np.isfinite(tot):
                nxt = int(np.random.choice(cand))
            else:
                p = pr / tot
                p = np.maximum(p, 0)
                if p.sum() > 0:
                    try:
                        nxt = int(np.random.choice(cand, p=p))
                    except ValueError:
                        nxt = int(np.random.choice(cand))
                else:
                    nxt = int(np.random.choice(cand))
            tour.append(nxt)
            vis[nxt] = True
            curr = nxt
        if tour[-1] != 0:
            tour.append(0)
        return tour

    def _construct_solutions(self) -> List[List[int]]:
        for i in range(self.num_ants):
            self.ant_tours[i] = self._build_tour()
        return self.ant_tours

    def _tour_length(self, tour: List[int]) -> float:
        if len(tour) < 2:
            return float('inf')
        arr = np.array(tour, dtype=np.int64)
        if USE_NUMBA:
            L = numba_tour_length(arr, self.dist)
            return L if L >= 0 else float('inf')
        return float(self.dist[arr[:-1], arr[1:]].sum())

    def _update_pheromones_python(self, tours):
        for t in tours:
            L = self._tour_length(t)
            if L <= 0 or not np.isfinite(L):
                continue
            delta = self.Q / L
            arr = np.array(t, dtype=np.int64)
            for i in range(len(arr) - 1):
                a, b = arr[i], arr[i + 1]
                if 0 <= a < self.N and 0 <= b < self.N:
                    self.pher[a, b] += delta
                    self.pher[b, a] += delta

    def _update_pheromones(self, tours):
        self.pher *= (1 - self.rho)
        if USE_NUMBA:
            try:
                valids = [np.array(t, dtype=np.int64) for t in tours if len(t) > 1]
                lst = numba.typed.List(valids)
                numba_update_pheromones_deposit(lst, self.dist, self.pher,
                                                self.Q, self.N)
            except:
                self._update_pheromones_python(tours)
        else:
            self._update_pheromones_python(tours)
        np.fill_diagonal(self.pher, 0.0)

    def run(self,
            iterations: int,
            dynamic_update_fn: Optional[Callable[["ACO_DVRP"], None]],
            dynamic_every: int,
            realtime_plot: bool = True
            ) -> Tuple[List[int], float, List[float], List[float]]:

        best_tour: List[int] = []
        best_len: float = float("inf")
        history: List[float] = []
        relatives: List[float] = []
        last_before: Optional[float] = None

        fig, ax = None, None
        plot_active = False
        if realtime_plot:
            try:
                fig, ax = setup_realtime_plot()
                plot_active = True
            except Exception as e:
                print(f"Warning: no real‑time plot: {e}")
                realtime_plot = False

        for it in range(1, iterations + 1):
            start = time.time()

            # dynamic injection
            if dynamic_update_fn and it % dynamic_every == 0:
                last_before = best_len if best_len < float("inf") else None
                print(f"→ Iter {it}: injecting new node…")
                dynamic_update_fn(self)
                oldN = self.N
                newN = self.dist.shape[0]
                if newN > oldN:
                    avg = self.pher[self.pher > 0].mean() \
                          if np.any(self.pher > 0) \
                          else INITIAL_PHEROMONE
                    P = np.full((newN, newN), avg, dtype=np.float64)
                    P[:oldN, :oldN] = self.pher
                    np.fill_diagonal(P, 0.0)
                    self.pher = P
                    with np.errstate(divide="ignore", invalid="ignore"):
                        self.eta = np.where(self.dist > 0,
                                            1.0 / self.dist, 0.0)
                    self.N = newN
                    if USE_CANDIDATE_LIST:
                        self._build_candidate_list(CANDIDATE_LIST_SIZE)
                    best_len, best_tour = float("inf"), []
                    self.ant_tours = [[] for _ in range(self.num_ants)]
                    if BENCHMARK_RELATIVE and last_before is not None:
                        relatives.append(np.nan)
                    print(f"  New N = {self.N}")

            # 1) build all tours
            tours = self._construct_solutions()

            # 2) selective polishing on top‑k raw tours
            tour_lens = [self._tour_length(t) for t in tours]
            k = max(MIN_TOP_K, int(TOP_K_FRACTION * self.num_ants))
            top_k_indices = sorted(range(self.num_ants), key=lambda i: tour_lens[i])[:k]
            for i in top_k_indices:
                tours[i] = two_opt_budgeted(tours[i], self.dist, BUDGETED_2OPT_TRIALS)

            # 3) update pheromones on the mixed set
            self._update_pheromones(tours)

            # select this-iteration best (lightly polished)
            it_best = float("inf")
            it_tour: List[int] = []
            for t in tours:
                L = self._tour_length(t)
                if L < it_best:
                    it_best, it_tour = L, t

            # full 2-opt on iteration best
            opt_tour = two_opt(it_tour, self.dist)
            it_best = tour_length(opt_tour, self.dist)
            it_tour = opt_tour

            # occasional 3-opt on global best
            if it % THREE_OPT_CADENCE == 0 and best_tour:
                candidate = three_opt(best_tour, self.dist)
                cand_len = tour_length(candidate, self.dist)
                if cand_len < best_len:
                    best_tour, best_len = candidate, cand_len
                    print(f"☆ Iter {it}: 3‑opt improved global → {best_len:.2f}")

            # update global best
            if it_best < best_len:
                best_len, best_tour = it_best, it_tour

            history.append(best_len)

            # realtime plot update
            if realtime_plot and plot_active and it % PLOT_UPDATE_FREQ == 0:
                assert ax is not None
                if self.coords is not None:
                    plot_active = update_realtime_plot(ax, self.coords, best_tour, it, best_len)
                else:
                    print("Warning: coords missing for plot.")

            # relative benchmark logging
            if BENCHMARK_RELATIVE and dynamic_update_fn \
               and it > 1 and (it - 1) % dynamic_every == 0:
                if last_before is not None and relatives \
                   and np.isnan(relatives[-1]):
                    relatives[-1] = (best_len - last_before) / last_before \
                                    if last_before > 0 else 0.0
                    last_before = None

            print(f"Iter {it}/{iterations} best={best_len:.2f} ({time.time() - start:.3f}s)")

        if realtime_plot:
            plt.ioff()
            if plot_active:
                ax.set_title(f"Final Route | Len {best_len:.2f}")
                plt.show(block=False)

        relatives = [r for r in relatives if not np.isnan(r)]
        return best_tour, best_len, history, relatives


def main():
    print("Generating initial map…")
    coords = generate_random_nodes(NUM_CUSTOMERS, X_BOUNDS, Y_BOUNDS)

    coords_tuple = tuple(map(tuple, coords))
    dist_mat = compute_distance_matrix(coords_tuple)

    print(f"Initial map: {coords.shape[0]} nodes.")

    print("\nInitializing ACO…")
    aco = ACO_DVRP(
        dist_matrix=dist_mat,
        num_ants=NUM_ANTS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        Q=Q,
        initial_pheromone=INITIAL_PHEROMONE
    )
    aco.coords = coords

    print(f"\nRunning ACO for {ITERATIONS} iterations…")
    start_time = time.time()
    best_tour, best_len, history, relatives = aco.run(
        iterations=ITERATIONS,
        dynamic_update_fn=dynamic_add_node,
        dynamic_every=DYNAMIC_EVERY,
        realtime_plot=ENABLE_REALTIME_PLOT
    )
    run_time = time.time() - start_time
    print(f"\n--- ACO finished in {run_time:.2f}s ---")

    print("\n== Final Result ==")
    if best_tour:
        print(f"Best tour (edges={len(best_tour) - 1}, N={aco.N}):")
        print(" -> ".join(map(str, best_tour)))
        print(f"Length: {best_len:.4f}")
    else:
        print("No valid tour found.")

    print("\nGenerating final plots…")
    if history:
        if BENCHMARK_RELATIVE and relatives:
            plot_convergence_and_relative(history, relatives, DYNAMIC_EVERY)
        else:
            plot_convergence(history)

    if BENCHMARK_RELATIVE and relatives and not history:
        plot_relative(relatives, DYNAMIC_EVERY)

    if aco.coords is not None:
        plot_route(aco.coords, best_tour,
                   title=f"Final Best Route (N={aco.N})",
                   runtime=run_time)

    if plt.get_fignums():
        print("Close plots to exit.")
        plt.show()
    else:
        print("No plots generated.")
    print("\nDone.")


if __name__ == "__main__":
    main()