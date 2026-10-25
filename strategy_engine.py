
"""Genetic Algorithm Strategy Engine for multi-UAV disaster-grid deployment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GRID_SIZE = 30
N_DRONES = 3
N_WAYPOINTS = 5
MISSION_STEPS = N_WAYPOINTS + 1
STARTS = [(2, 2), (2, 27), (27, 2)]
PRIORITY_TARGETS = [(8, 8), (15, 15), (22, 20), (23, 7), (7, 23)]
NO_FLY = [(11, 5, 16, 12), (17, 17, 22, 24)]  # inclusive rectangles
HAZARDS = [(5, 15, 11, 21, 3.0), (20, 3, 26, 10, 2.0), (14, 22, 18, 28, 1.5)]

@dataclass
class DronePlan:
    """One drone's waypoint coordinates and sensor-power schedule."""
    waypoints: np.ndarray  # shape (N_WAYPOINTS, 2), normalized to [0, 1]
    sensor_power: np.ndarray  # shape (N_WAYPOINTS,), normalized to [0, 1]
    speed: np.ndarray  # shape (N_WAYPOINTS,), normalized to [0, 1]

@dataclass
class Individual:
    plans: List[DronePlan]
    fitness: float = float("-inf")
    objectives: dict | None = None

def decode_coord(v: float) -> int:
    return int(np.clip(round(float(v) * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))

def in_rect(x: int, y: int, rect) -> bool:
    x1, y1, x2, y2 = rect[:4]
    return x1 <= x <= x2 and y1 <= y <= y2

def is_no_fly(x: int, y: int) -> bool:
    return any(in_rect(x, y, r) for r in NO_FLY)

def hazard_cost(x: int, y: int) -> float:
    return sum(r[4] for r in HAZARDS if in_rect(x, y, r))

def bresenham(a: Tuple[int, int], b: Tuple[int, int]):
    """Grid cells along a straight segment; used to score traversed risk/obstacles."""
    x0, y0 = a; x1, y1 = b
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy

def make_individual(rng=random) -> Individual:
    plans = []
    for _ in range(N_DRONES):
        # Mixed real-valued chromosome: waypoint x/y, sensor power, and speed.
        plans.append(DronePlan(
            waypoints=np.array([[rng.random(), rng.random()] for _ in range(N_WAYPOINTS)]),
            sensor_power=np.array([rng.uniform(.25, 1.0) for _ in range(N_WAYPOINTS)]),
            speed=np.array([rng.uniform(.25, 1.0) for _ in range(N_WAYPOINTS)])))
    return Individual(plans)

def evaluate(ind: Individual, weights=None) -> float:
    """Weighted scalar fitness with explicit penalties for hard constraints.
    Higher is better. Coverage/detection are rewards; energy/time/risk/violations costs.
    """
    if weights is None:
        weights = dict(coverage=8.0, detection=5.0, energy=1.5, time=1.0, risk=2.5,
                       nofly=18.0, range=10.0, overlap=1.0)
    covered = set()
    detection_score = 0.0
    energy = 0.0
    flight_time = 0.0
    risk = 0.0
    nofly_violations = 0
    range_excess = 0.0
    route_cells = []
    for di, plan in enumerate(ind.plans):
        points = [STARTS[di]]
        points += [(decode_coord(x), decode_coord(y)) for x, y in plan.waypoints]
        prev = points[0]
        route_length = 0.0
        for j, p in enumerate(points[1:]):
            x, y = p
            segment = list(bresenham(prev, p))
            route_length += math.dist(prev, p)
            for cx, cy in segment:
                route_cells.append((cx, cy))
                if is_no_fly(cx, cy):
                    nofly_violations += 1
                risk += hazard_cost(cx, cy)
                # Coverage is a radius-2 sensor footprint at each waypoint.
            power = float(plan.sensor_power[j])
            speed = .5 + float(plan.speed[j]) * 1.5
            for tx, ty in PRIORITY_TARGETS:
                if math.dist((x, y), (tx, ty)) <= 3.0:
                    detection_score += power
            for cx in range(max(0, x-2), min(GRID_SIZE, x+3)):
                for cy in range(max(0, y-2), min(GRID_SIZE, y+3)):
                    if math.dist((x, y), (cx, cy)) <= 2.5 and not is_no_fly(cx, cy):
                        covered.add((cx, cy))
            energy += (0.6 + 0.9 * power + 0.4 * speed) * math.dist(prev, p)
            flight_time += math.dist(prev, p) / speed
            prev = p
        # Range limit of 65 grid-distance units per drone.
        range_excess += max(0.0, route_length - 65.0)
    coverage_ratio = len(covered) / float(GRID_SIZE * GRID_SIZE)
    target_detection = min(1.0, detection_score / max(1, len(PRIORITY_TARGETS)))
    overlap = max(0, len(route_cells) - len(set(route_cells)))
    score = (weights["coverage"] * coverage_ratio * 100
             + weights["detection"] * target_detection * 10
             - weights["energy"] * energy / 10
             - weights["time"] * flight_time / 5
             - weights["risk"] * risk
             - weights["nofly"] * nofly_violations
             - weights["range"] * range_excess
             - weights["overlap"] * overlap / 3)
    ind.fitness = float(score)
    ind.objectives = dict(coverage=coverage_ratio, detection=target_detection, energy=energy,
                          time=flight_time, risk=risk, nofly_violations=nofly_violations,
                          range_excess=range_excess)
    return ind.fitness

def tournament(population, k=3):
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda i: i.fitness)

def crossover(a: Individual, b: Individual, rate=.85) -> Individual:
    """Segment-wise crossover; swap a drone's complete waypoint/sensor/speed segment."""
    child_plans = []
    for pa, pb in zip(a.plans, b.plans):
        if random.random() < rate:
            cut = random.randint(1, N_WAYPOINTS-1)
            wp = np.vstack([pa.waypoints[:cut], pb.waypoints[cut:]])
            sp = np.concatenate([pa.sensor_power[:cut], pb.sensor_power[cut:]])
            sv = np.concatenate([pa.speed[:cut], pb.speed[cut:]])
        else:
            wp, sp, sv = pa.waypoints.copy(), pa.sensor_power.copy(), pa.speed.copy()
        child_plans.append(DronePlan(wp, sp, sv))
    return Individual(child_plans)

def mutate(ind: Individual, rate=.12, scale=.12):
    """Gaussian coordinate mutation plus bounded sensor/speed mutation."""
    for p in ind.plans:
        for i in range(N_WAYPOINTS):
            if random.random() < rate:
                p.waypoints[i] += np.random.normal(0, scale, 2)
            if random.random() < rate:
                p.sensor_power[i] += random.gauss(0, scale)
            if random.random() < rate:
                p.speed[i] += random.gauss(0, scale)
        p.waypoints[:] = np.clip(p.waypoints, 0, 1)
        p.sensor_power[:] = np.clip(p.sensor_power, .1, 1.0)
        p.speed[:] = np.clip(p.speed, .1, 1.0)

def run_ga(pop_size=70, generations=100, mutation_rate=.12, tournament_k=3, seed=42):
    random.seed(seed); np.random.seed(seed)
    pop = [make_individual() for _ in range(pop_size)]
    for ind in pop: evaluate(ind)
    best_history, avg_history = [], []
    best = max(pop, key=lambda i: i.fitness)
    for _ in range(generations):
        pop.sort(key=lambda i: i.fitness, reverse=True)
        if pop[0].fitness > best.fitness:
            best = pop[0]
        best_history.append(pop[0].fitness)
        avg_history.append(float(np.mean([x.fitness for x in pop])))
        next_pop = [clone(pop[0]), clone(pop[1])]  # elitism
        while len(next_pop) < pop_size:
            p1, p2 = tournament(pop, tournament_k), tournament(pop, tournament_k)
            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            evaluate(child)
            next_pop.append(child)
        pop = next_pop
    pop.sort(key=lambda i: i.fitness, reverse=True)
    if pop[0].fitness > best.fitness: best = pop[0]
    return best, best_history, avg_history

def clone(ind):
    return Individual([DronePlan(p.waypoints.copy(), p.sensor_power.copy(), p.speed.copy()) for p in ind.plans],
                      ind.fitness, dict(ind.objectives) if ind.objectives else None)

def run_random_search(samples=7000, seed=99):
    random.seed(seed); np.random.seed(seed)
    best = None
    for _ in range(samples):
        x = make_individual(); evaluate(x)
        if best is None or x.fitness > best.fitness: best = x
    return best

def plot_solution(ind, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-.5, GRID_SIZE-.5); ax.set_ylim(-.5, GRID_SIZE-.5)
    ax.set_xticks(range(0, GRID_SIZE, 5)); ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.grid(True, alpha=.2)
    for x1,y1,x2,y2 in NO_FLY:
        ax.add_patch(plt.Rectangle((x1-.5,y1-.5), x2-x1+1, y2-y1+1, color="red", alpha=.2))
    for x1,y1,x2,y2,w in HAZARDS:
        ax.add_patch(plt.Rectangle((x1-.5,y1-.5), x2-x1+1, y2-y1+1, color="orange", alpha=.22))
    for tx,ty in PRIORITY_TARGETS: ax.scatter(tx,ty,marker="*",s=130,color="black")
    for i,p in enumerate(ind.plans):
        pts=[STARTS[i]]+[(decode_coord(x),decode_coord(y)) for x,y in p.waypoints]
        xs=[v[0] for v in pts]; ys=[v[1] for v in pts]
        ax.plot(xs,ys,marker="o",linewidth=2,label=f"UAV {i+1}")
        for x,y in pts[1:]:
            circ=plt.Circle((x,y),2.5,fill=False,alpha=.25)
            ax.add_patch(circ)
    ax.set_title("Best evolved UAV routes (red: no-fly; orange: hazard; stars: targets)")
    ax.set_xlabel("Grid X"); ax.set_ylabel("Grid Y"); ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def plot_convergence(best, avg, path):
    fig, ax=plt.subplots(figsize=(8,4.5))
    ax.plot(range(1,len(best)+1),best,label="Generation best")
    ax.plot(range(1,len(avg)+1),avg,label="Population average")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (higher is better)")
    ax.set_title("GA fitness convergence"); ax.grid(True,alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def plot_coverage(ind, path):
    heat=np.zeros((GRID_SIZE,GRID_SIZE))
    for p in ind.plans:
        for x,y in p.waypoints:
            cx,cy=decode_coord(x),decode_coord(y)
            for i in range(max(0,cx-2),min(GRID_SIZE,cx+3)):
                for j in range(max(0,cy-2),min(GRID_SIZE,cy+3)):
                    if math.dist((cx,cy),(i,j))<=2.5 and not is_no_fly(i,j): heat[j,i]+=1
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(heat,origin="lower",cmap="YlOrRd",interpolation="nearest")
    fig.colorbar(im,ax=ax,label="Number of UAV sensor footprints")
    ax.set_title("Coverage heatmap from evolved waypoints"); ax.set_xlabel("Grid X"); ax.set_ylabel("Grid Y")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

if __name__ == "__main__":
    out=Path("outputs"); out.mkdir(exist_ok=True)
    best,bh,ah=run_ga()
    plot_convergence(bh,ah,out/"fitness_convergence.png")
    plot_solution(best,out/"evolved_routes.png")
    plot_coverage(best,out/"coverage_heatmap.png")
    random_best=run_random_search(samples=7000)
    with open(out/"metrics.json","w") as f:
        json.dump({"ga_fitness":best.fitness,"ga_objectives":best.objectives,
                   "random_search_fitness":random_best.fitness,
                   "best_history":bh,"average_history":ah},f,indent=2)
    print("GA best fitness:", round(best.fitness,3), best.objectives)
    print("Random-search best fitness:", round(random_best.fitness,3))
