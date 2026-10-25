# Multi-UAV Disaster Grid Strategy Engine (Genetic Algorithm)

A runnable educational prototype for the Artificial Intelligence Assignment 2. It optimizes three UAV routes, sensor-power schedules, and speed profiles on a discretized 30×30 grid.

## Features
- Mixed real-valued chromosome: waypoint `(x,y)` pairs, sensor power, and speed for each drone.
- Scalar multi-objective fitness: coverage and target detection rewards; energy, time, hazard exposure, route overlap, no-fly violations, and excess range penalties.
- Tournament selection, elitism, segment-wise crossover, and bounded Gaussian mutation.
- Hard-constraint penalties for no-fly cells and route range. The score discourages violations; this prototype does not guarantee every line segment is repaired to feasibility.
- Visualizations: fitness convergence, evolved routes, and waypoint coverage heatmap.
- Random-search baseline.

## Requirements
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From the project root:

```bash
python -m uav_ga.strategy_engine
```

Outputs are written to `outputs/`: `fitness_convergence.png`, `evolved_routes.png`, `coverage_heatmap.png`, and `metrics.json`.

## Model assumptions
- Grid cells are integer `(x,y)` coordinates on a 30×30 grid.
- Three UAVs start at fixed corners; each has five evolved waypoints.
- Sensor footprint radius is 2.5 grid units; target detection radius is 3 units.
- The no-fly rectangles and hazard rectangles are synthetic demonstration data, not real maps.
- Straight segments between waypoints are scored using a Bresenham grid traversal. For operational deployment, use validated geospatial data, robust collision checking, vehicle dynamics, uncertainty modeling, and safety-critical route repair.

## Fitness
Higher is better:
`F = 8(100 C) + 5(10 D) - 1.5 E/10 - T/5 - 2.5 R - 18 N - 10 X - O/3`
where `C` is covered grid-cell fraction, `D` is normalized target-detection score, `E` is energy proxy, `T` is flight-time proxy, `R` is accumulated hazard cost, `N` is the count of no-fly traversal violations, `X` is distance beyond the per-drone 65-unit range, and `O` is route-cell overlap count. These weights are tunable and express illustrative priorities, not universal values.

## Repository structure
```
uav_ga_assignment/
├── uav_ga/
│   ├── __init__.py
│   └── strategy_engine.py
├── outputs/
│   ├── fitness_convergence.png
│   ├── evolved_routes.png
│   ├── coverage_heatmap.png
│   └── metrics.json
├── report/
│   └── technical_report.docx
├── requirements.txt
└── README.md
```

## Reproducibility
Default GA seed is 42 and random-search seed is 99. The objective is stochastic, so results may change with parameters and seeds. This is a simulation prototype, not a certified autonomous flight planner.
