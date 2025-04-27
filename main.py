#!/usr/bin/env python3
"""
Dynamic Vehicle Routing Problem using Ant Colony Optimization
– with a fixed SEED for reproducible initial map and dynamic injections,
  a static real‑time plotting interval of 50 iterations,
"""

import random
import time  # For timing Numba compilation and overall runtime
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ——————————————————————————————————————————————————————————————————
#                          RANDOM SEEDING
# ——————————————————————————————————————————————————————————————————
# Change this SEED to reproduce the same initial customer map
# and the same injected customers on every run.
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
print(f"Using fixed SEED={SEED}")

# ——————————————————————————————————————————————————————————————————
#                      HYPERPARAMETERS & TOGGLES
# ——————————————————————————————————————————————————————————————————
NUM_CUSTOMERS       = 25
X_BOUNDS            = (0, 50)
Y_BOUNDS            = (0, 50)

NUM_ANTS            = 70
ALPHA, BETA         = 1.0, 3.0
RHO, Q              = 0.4, 120.0
INITIAL_PHEROMONE   = 1.0

USE_CANDIDATE_LIST  = True
CANDIDATE_LIST_SIZE = 10

ITERATIONS          = 2000
DYNAMIC_EVERY       = 300

BENCHMARK_RELATIVE  = True

ENABLE_REALTIME_PLOT = True

FIGSIZE_ROUTE    = (7, 7)
FIGSIZE_CONV     = (8, 4)
FIGSIZE_RELATIVE = (8, 4)

# ——————————————————————————————————————————————————————————————————
#                        NUMBA CONFIGURATION
# ——————————————————————————————————————————————————————————————————
try:
    import numba
    USE_NUMBA = True
    print("Numba detected. Will use JIT compilation where enabled.")
except ImportError:
    numba = None
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    class DummyNumba:
        pass
    numba = DummyNumba()
    numba.njit = njit
    USE_NUMBA = False
    print("Numba not found. Running pure Python/NumPy code.")

# ——————————————————————————————————————————————————————————————————
#                        NUMBA‑ACCELERATED HELPERS
# ——————————————————————————————————————————————————————————————————
@numba.njit(cache=True)
def numba_weighted_choice(candidates: np.ndarray, probs: np.ndarray) -> int:
    s = probs.sum()
    if s <= 0:
        return candidates[np.random.randint(0, len(candidates))] if len(candidates) > 0 else -1
    cum = np.cumsum(probs)
    r = np.random.rand()
    idx = np.searchsorted(cum, r, side='right')
    return candidates[min(idx, len(candidates) - 1)]

@numba.njit(cache=True)
def numba_tour_length(tour_arr: np.ndarray, dist_matrix: np.ndarray) -> float:
    total = 0.0
    N = dist_matrix.shape[0]
    for i in range(len(tour_arr) - 1):
        u, v = tour_arr[i], tour_arr[i+1]
        if 0 <= u < N and 0 <= v < N:
            total += dist_matrix[u, v]
        else:
            return -1.0
    return total

@numba.njit(cache=True)
def numba_update_pheromones_deposit(
    tours_list: numba.typed.List,
    dist_matrix: np.ndarray,
    pher_matrix: np.ndarray,
    Q: float,
    N: int
):
    for tour_arr in tours_list:
        L = numba_tour_length(tour_arr, dist_matrix)
        if L <= 0 or np.isnan(L) or np.isinf(L):
            continue
        delta = Q / L
        for j in range(len(tour_arr) - 1):
            a, b = tour_arr[j], tour_arr[j+1]
            if 0 <= a < N and 0 <= b < N:
                pher_matrix[a, b] += delta
                pher_matrix[b, a] += delta

# ——————————————————————————————————————————————————————————————————
#                           PLOTTING HELPERS
# ——————————————————————————————————————————————————————————————————
def setup_realtime_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=FIGSIZE_ROUTE)
    ax.set_title("ACO‑DVRP Real‑time Route")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(0.1)
    return fig, ax

def update_realtime_plot(
    ax, coords: np.ndarray, tour: List[int],
    iteration: int, best_len: float
) -> bool:
    if not plt.fignum_exists(ax.figure.number):
        print("Real‑time plot window closed. Stopping updates.")
        return False

    ax.clear()
    xs, ys = coords[:, 0], coords[:, 1]

    if coords.shape[0] > 1:
        ax.scatter(xs[1:], ys[1:], c="red", marker="o",
                   label="Customers", zorder=3, s=25)
    ax.scatter(xs[0], ys[0], c="blue", marker="s",
               label="Depot", zorder=3, s=100)

    if tour and len(tour) > 1:
        valid = [i for i in tour if 0 <= i < coords.shape[0]]
        if len(valid) == len(tour):
            ax.plot(coords[valid, 0], coords[valid, 1],
                    "-o", c="lightblue", markersize=4,
                    linewidth=1.5, label="Route", zorder=2)
        else:
            print(f"Warning (Plot): invalid indices in tour {tour}.")

    ax.set_title(f"Iteration {iteration} | Best Length: {best_len:.2f}")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axis('equal')
    ax.figure.canvas.draw_idle()
    plt.pause(0.01)
    return True

def plot_convergence(history: List[float]):
    plt.figure(figsize=FIGSIZE_CONV)
    plt.plot(history, marker='.', linestyle='-', color='blue', markersize=4)
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title("ACO‑DVRP Convergence")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_relative(relatives: List[float], dynamic_every: int):
    if not relatives:
        print("No relative increase data to plot.")
        return
    plt.figure(figsize=FIGSIZE_RELATIVE)
    iters = (np.arange(len(relatives)) + 1) * dynamic_every
    plt.bar(iters, relatives, width=dynamic_every * 0.8,
            color="orange", edgecolor="k")
    plt.xlabel("Iteration of Injection")
    plt.ylabel("Relative Increase")
    plt.title("Relative Tour‑Length Increase per Injection")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_route(
    coords: np.ndarray,
    tour: List[int],
    title: str = "Final Best Route",
    runtime: Optional[float] = None
):
    if coords is None or coords.shape[0] == 0:
        print("No coordinates to plot for final route.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_ROUTE)
    xs, ys = coords[:, 0], coords[:, 1]

    if coords.shape[0] > 1:
        ax.scatter(xs[1:], ys[1:], c="red", marker="o",
                   label="Customers", zorder=3, s=25)
    ax.scatter(xs[0], ys[0], c="blue", marker="s",
               label="Depot", zorder=3, s=100)

    if tour:
        valid = [i for i in tour if 0 <= i < coords.shape[0]]
        if len(valid) == len(tour):
            ax.plot(coords[valid, 0], coords[valid, 1],
                    "-o", c="lightblue", markersize=4,
                    linewidth=1.5, label="Route", zorder=2)
        else:
            print(f"Warning (Final Plot): invalid indices in tour {tour}.")

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.axis("equal")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)

    # stamp total runtime in the top‑left corner
    if runtime is not None:
        ax.text(
            0.02, 0.98,
            f"Total runtime: {runtime:.2f}s",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

    plt.tight_layout()

# ——————————————————————————————————————————————————————————————————
#                       DATA GENERATION & DYNAMIC UPDATE
# ——————————————————————————————————————————————————————————————————
def generate_random_nodes(
    num_customers: int,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float]
) -> np.ndarray:
    depot = np.array([[0.0, 0.0]])
    xs = np.random.uniform(x_bounds[0], x_bounds[1], size=(num_customers, 1))
    ys = np.random.uniform(y_bounds[0], y_bounds[1], size=(num_customers, 1))
    return np.vstack([depot, np.hstack([xs, ys])])

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))

def dynamic_add_node(aco: "ACO_DVRP"):
    # Increment and reseed per injection for reproducibility
    aco._injection_count += 1
    rnd = SEED + aco._injection_count
    random.seed(rnd)
    np.random.seed(rnd)

    xmin, xmax = X_BOUNDS
    ymin, ymax = Y_BOUNDS
    new_pt = np.array([[random.uniform(xmin, xmax),
                        random.uniform(ymin, ymax)]])
    aco.coords = np.vstack([aco.coords, new_pt])
    aco.dist = compute_distance_matrix(aco.coords)

# ——————————————————————————————————————————————————————————————————
#                              ACO CLASS
# ——————————————————————————————————————————————————————————————————
class ACO_DVRP:
    def __init__(
        self,
        dist_matrix: np.ndarray,
        num_ants: int,
        alpha: float,
        beta: float,
        rho: float,
        Q: float,
        initial_pheromone: float,
    ):
        # Validate distance matrix
        if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("dist_matrix must be a square 2D array")
        if np.any(dist_matrix < 0):
            raise ValueError("Distances cannot be negative")

        self.dist = dist_matrix.copy()
        self.N = self.dist.shape[0]
        self.pher = np.full((self.N, self.N), initial_pheromone, dtype=np.float64)
        np.fill_diagonal(self.pher, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            eta = 1.0 / self.dist
        eta[~np.isfinite(eta)] = 0.0
        self.eta = eta.astype(np.float64)

        self.num_ants = num_ants
        self.alpha    = alpha
        self.beta     = beta
        self.rho      = rho
        self.Q        = Q

        # for dynamic injections
        self._injection_count = 0

        # candidate‐list initialization
        self.neighbors_list: Optional[List[np.ndarray]] = None
        if USE_CANDIDATE_LIST:
            self._build_candidate_list(CANDIDATE_LIST_SIZE)

        self.coords: Optional[np.ndarray] = None

        # pre‑compile Numba functions if available
        if USE_NUMBA:
            print("Pre‑compiling Numba functions (may take a moment)…")
            start_time = time.time()
            try:
                # Dummy data for compilation
                _dummy_tour = np.array([0,0], dtype=np.int64)
                _dummy_dist = self.dist[:2,:2].copy() if self.N>1 else np.array([[0.]])
                _dummy_pher = self.pher[:2,:2].copy() if self.N>1 else np.array([[0.]])
                _dummy_eta  = self.eta[:2,:2].copy() if self.N>1 else np.array([[0.]])
                _dummy_vis  = np.zeros(self.N, dtype=np.bool_)
                nbrs_data   = [np.array([1],dtype=np.int64)]*self.N if self.N>1 else [np.array([0],dtype=np.int64)]
                _dummy_nbrs = numba.typed.List(nbrs_data)

                _ = self._build_tour_numba(0, _dummy_vis.copy(), self.N,
                                           _dummy_pher, _dummy_eta,
                                           self.alpha, self.beta,
                                           USE_CANDIDATE_LIST, _dummy_nbrs)
                _ = numba_tour_length(_dummy_tour, _dummy_dist)
                tours_list = numba.typed.List([_dummy_tour])
                _ = numba_update_pheromones_deposit(
                    tours_list, _dummy_dist, _dummy_pher, self.Q, _dummy_dist.shape[0]
                )
                print(f"Numba compilation finished in {(time.time()-start_time):.2f}s.")
            except Exception as e:
                print(f"Numba compile warning: {e}")

    def _build_candidate_list(self, k: int):
        k = min(k, self.N - 1)
        if k <= 0:
            self.neighbors_list = None
            return
        self.neighbors_list = numba.typed.List()
        order = np.argsort(self.dist, axis=1)
        for i in range(self.N):
            neighbors = order[i, 1:k+1].astype(np.int64)
            self.neighbors_list.append(neighbors)

    @staticmethod
    @numba.njit(cache=True)
    def _build_tour_numba(
        start_node: int,
        visited_mask: np.ndarray,
        N: int,
        pher_matrix: np.ndarray,
        eta_matrix: np.ndarray,
        alpha: float,
        beta: float,
        use_cand_list: bool,
        neighbors_list: numba.typed.List
    ) -> np.ndarray:
        tour = np.empty(N+1, dtype=np.int64)
        tour[0] = start_node
        visited_mask[start_node] = True
        current = start_node
        idx = 1
        for _ in range(N-1):
            if use_cand_list and current < len(neighbors_list):
                cand = neighbors_list[current]
                mask = ~visited_mask[cand]
                candidates = cand[mask]
                if len(candidates) == 0:
                    candidates = np.arange(N)[~visited_mask]
            else:
                candidates = np.arange(N)[~visited_mask]

            if len(candidates) == 0:
                break

            pher = pher_matrix[current, candidates]
            eta  = eta_matrix[current, candidates]
            num  = (pher**alpha) * (eta**beta)
            total = np.sum(num)
            if total <= 0 or not np.isfinite(total):
                nxt = candidates[np.random.randint(0, len(candidates))]
            else:
                probs = num / total
                nxt = numba_weighted_choice(candidates, probs)
                if nxt == -1:
                    break

            tour[idx] = nxt
            visited_mask[nxt] = True
            current = nxt
            idx += 1

        if idx == N:
            tour[N] = start_node
            return tour

        final = np.empty(idx+1, dtype=np.int64)
        final[:idx] = tour[:idx]
        final[idx] = start_node
        return final

    def _build_tour(self) -> List[int]:
        if USE_NUMBA and self.neighbors_list is not None:
            visited = np.zeros(self.N, dtype=np.bool_)
            arr = self._build_tour_numba(
                0, visited, self.N, self.pher, self.eta,
                self.alpha, self.beta,
                True, self.neighbors_list
            )
            return arr.tolist()

        visited = np.zeros(self.N, dtype=bool)
        visited[0] = True
        tour = [0]
        current = 0
        for _ in range(self.N-1):
            if USE_CANDIDATE_LIST and self.neighbors_list and current < len(self.neighbors_list):
                cand = self.neighbors_list[current]
                mask = ~visited[cand]
                candidates = cand[mask]
                if candidates.size == 0:
                    candidates = np.where(~visited)[0]
            else:
                candidates = np.where(~visited)[0]

            if candidates.size == 0:
                break

            pher = self.pher[current, candidates]
            eta  = self.eta[current, candidates]
            num  = (pher**self.alpha) * (eta**self.beta)
            total = num.sum()
            if total <= 0 or not np.isfinite(total):
                nxt = int(np.random.choice(candidates))
            else:
                probs = num / total
                probs = np.maximum(probs, 0)
                s = probs.sum()
                if s <= 0:
                    nxt = int(np.random.choice(candidates))
                else:
                    try:
                        nxt = int(np.random.choice(candidates, p=probs))
                    except ValueError:
                        nxt = int(np.random.choice(candidates))
            tour.append(nxt)
            visited[nxt] = True
            current = nxt

        if tour[-1] != 0:
            tour.append(0)
        return tour

    def _construct_solutions(self) -> List[List[int]]:
        return [self._build_tour() for _ in range(self.num_ants)]

    def _tour_length(self, tour: List[int]) -> float:
        if not tour or len(tour) < 2:
            return float('inf')
        arr = np.array(tour, dtype=np.int64)
        if USE_NUMBA:
            L = numba_tour_length(arr, self.dist)
            return L if L >= 0 else float('inf')
        if np.any(arr < 0) or np.any(arr >= self.N):
            return float('inf')
        return float(self.dist[arr[:-1], arr[1:]].sum())

    def _update_pheromones_python(self, tours: List[List[int]]):
        for t in tours:
            L = self._tour_length(t)
            if L <= 0 or not np.isfinite(L):
                continue
            delta = self.Q / L
            arr = np.array(t, dtype=np.int64)
            for i in range(len(arr)-1):
                a, b = arr[i], arr[i+1]
                if 0 <= a < self.N and 0 <= b < self.N:
                    self.pher[a,b] += delta
                    self.pher[b,a] += delta

    def _update_pheromones(self, tours: List[List[int]]):
        self.pher *= (1 - self.rho)
        if USE_NUMBA:
            try:
                valid = [np.array(t, dtype=np.int64) for t in tours if len(t)>1]
                typed = numba.typed.List(valid)
                numba_update_pheromones_deposit(typed, self.dist, self.pher, self.Q, self.N)
            except Exception:
                self._update_pheromones_python(tours)
        else:
            self._update_pheromones_python(tours)
        np.fill_diagonal(self.pher, 0.0)

    def run(
        self,
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
                print(f"Warning: cannot init real‑time plot: {e}")
                realtime_plot = False

        for it in range(1, iterations+1):
            start = time.time()
            # dynamic injection
            if dynamic_update_fn and it % dynamic_every == 0:
                last_before = best_len if best_len < float("inf") else None
                print(f"→ Iter {it}: injecting new node…")
                dynamic_update_fn(self)
                oldN = self.N
                newN = self.dist.shape[0]
                if newN > oldN:
                    avg = self.pher[self.pher>0].mean() if np.any(self.pher>0) else INITIAL_PHEROMONE
                    P = np.full((newN,newN), avg, dtype=np.float64)
                    P[:oldN,:oldN] = self.pher
                    np.fill_diagonal(P, 0.0)
                    self.pher = P
                    with np.errstate(divide="ignore", invalid="ignore"):
                        eta = 1.0/self.dist
                    eta[~np.isfinite(eta)] = 0.0
                    self.eta = eta
                    self.N = newN
                    if USE_CANDIDATE_LIST:
                        self._build_candidate_list(CANDIDATE_LIST_SIZE)
                    best_len, best_tour = float("inf"), []
                    if BENCHMARK_RELATIVE and last_before is not None:
                        relatives.append(np.nan)
                    print(f"  New N = {self.N}")

            tours = self._construct_solutions()
            self._update_pheromones(tours)

            it_best = float("inf"); it_tour: List[int] = []
            for t in tours:
                L = self._tour_length(t)
                if L < it_best:
                    it_best, it_tour = L, t
            if it_best < best_len:
                best_len, best_tour = it_best, it_tour

            history.append(best_len)

            # static real‑time update every 50 iterations
            if realtime_plot and plot_active and (it % 50 == 0):
                if self.coords is not None:
                    plot_active = update_realtime_plot(ax, self.coords, best_tour, it, best_len)
                else:
                    print("Warning (Plot): coords missing.")

            # record relative increase after injections
            if BENCHMARK_RELATIVE and dynamic_update_fn and it>1 and (it-1)%dynamic_every==0:
                if last_before is not None and relatives and np.isnan(relatives[-1]):
                    rel = ((best_len - last_before)/last_before) if last_before>0 else 0.0
                    relatives[-1] = rel
                    last_before = None

            print(f"Iter {it}/{iterations} best={best_len:.2f} ({time.time()-start:.3f}s)")

        if realtime_plot:
            plt.ioff()
            if plot_active:
                ax.set_title(f"Final Route (Iter {iterations}) | Length: {best_len:.2f}")
                plt.show(block=False)

        relatives = [r for r in relatives if not np.isnan(r)]
        return best_tour, best_len, history, relatives

def main():
    print("Generating initial map…")
    coords = generate_random_nodes(NUM_CUSTOMERS, X_BOUNDS, Y_BOUNDS)
    dist_mat = compute_distance_matrix(coords)
    print(f"Initial map: {coords.shape[0]} nodes (1 depot + {coords.shape[0]-1} customers).")

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
    print(f" - Ants: {NUM_ANTS}, α={ALPHA}, β={BETA}, ρ={RHO}, Q={Q}")
    print(f" - Dynamic injection every {DYNAMIC_EVERY} iters")
    print(f" - Benchmark relative={BENCHMARK_RELATIVE}")
    print(f" - Real‑time plot updates every 50 iterations")

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
        print(f"Best tour ({len(best_tour)-1} edges, N={aco.N}):")
        print(" -> ".join(map(str, best_tour)))
        print(f"Length: {best_len:.4f}")
    else:
        print("No valid tour found.")

    print("\nGenerating final plots…")
    if history:
        plot_convergence(history)
    if BENCHMARK_RELATIVE:
        plot_relative(relatives, DYNAMIC_EVERY)
    if aco.coords is not None:
        # pass total runtime to stamp on the final route figure
        plot_route(
            aco.coords,
            best_tour,
            title=f"Final Best Route (N={aco.N})",
            runtime=run_time
        )

    if plt.get_fignums():
        print("Displaying final plots. Close windows to exit.")
        plt.show()
    else:
        print("No plots generated.")

    print("\nDone.")

if __name__ == "__main__":
    main()