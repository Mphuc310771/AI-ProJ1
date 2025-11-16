# Makefile for Project_Swarm
# Place this file at Project_Swarm/Makefile

PY := python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip
PYV := $(VENV_DIR)/bin/python
RUN := $(shell if [ -x "$(PYV)" ]; then echo "$(PYV)"; else echo "$(PY)"; fi)

ARGS ?=

.PHONY: help venv install clean \
        aco-discrete aco-continuous \
        pso-discrete pso-continuous \
        abc-discrete abc-continuous \
        fa-discrete fa-continuous \
        cs-discrete cs-continuous \
        bfs dfs dijkstra \
        compare-discrete compare-continuous compare-traditional

help:
	@echo "Usage: make <target> [ARGS='...']"
	@echo ""
	@echo "Common targets:"
	@echo "  aco-discrete        run swarm_algorithms/ACO/discrete/main.py"
	@echo "  aco-continuous      run swarm_algorithms/ACO/continuous/main.py"
	@echo "  pso-discrete        run PSO discrete (if implemented)"
	@echo "  pso-continuous      run PSO continuous (if implemented)"
	@echo "  abc-*, fa-*, cs-*   run ABC/FA/CS folders (if implemented)"
	@echo "  bfs, dfs, dijkstra  run traditional algorithms"
	@echo "  compare-*           run experiments/*.py"
	@echo ""
	@echo "Examples:"
	@echo "  make aco-discrete ARGS='--dataset data/discrete/dintzig42_d.txt'"
	@echo "  make install"
	@echo "  make venv"

venv:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PY) -m venv $(VENV_DIR)
	$(PIP) install -U pip setuptools wheel

install: venv
	@echo "Installing requirements from requirements.txt..."
	$(PIP) install -r requirements.txt

clean:
	@echo "Cleaning python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

############################
# ACO
############################
aco-discrete:
	@echo "Running ACO (discrete)..."
	$(RUN) swarm_algorithms/ACO/discrete/main.py $(ARGS)

aco-continuous:
	@echo "Running ACO (continuous)..."
	$(RUN) swarm_algorithms/ACO/continuous/main.py $(ARGS)

############################
# PSO (placeholders: update paths if you add mains)
############################
pso-discrete:
	@echo "Running PSO (discrete)..."
	$(RUN) swarm_algorithms/PSO/discrete/main.py $(ARGS)

pso-continuous:
	@echo "Running PSO (continuous)..."
	$(RUN) swarm_algorithms/PSO/continuous/main.py $(ARGS)

############################
# ABC / FA / CS (placeholders)
############################
abc-discrete:
	@echo "Running ABC (discrete)..."
	$(RUN) swarm_algorithms/ABC/discrete/main.py $(ARGS)

abc-continuous:
	@echo "Running ABC (continuous)..."
	$(RUN) swarm_algorithms/ABC/continuous/main.py $(ARGS)

fa-discrete:
	@echo "Running FA/FL (discrete)..."
	$(RUN) swarm_algorithms/FA/FL/discrete/main.py $(ARGS)

fa-continuous:
	@echo "Running FA/FL (continuous)..."
	$(RUN) swarm_algorithms/FA/FL/continuous/main.py $(ARGS)

cs-discrete:
	@echo "Running CS (discrete)..."
	$(RUN) swarm_algorithms/CS/discrete/main.py $(ARGS)

cs-continuous:
	@echo "Running CS (continuous)..."
	$(RUN) swarm_algorithms/CS/continuous/main.py $(ARGS)

############################
# Traditional algorithms
############################
bfs:
	@echo "Running BFS..."
	$(RUN) traditional_algorithms/BFS/bfs.py $(ARGS)

dfs:
	@echo "Running DFS..."
	$(RUN) traditional_algorithms/DFS/dfs.py $(ARGS)

dijkstra:
	@echo "Running Dijkstra..."
	$(RUN) traditional_algorithms/Dijkstra/dijkstra.py $(ARGS)

############################
# Experiments / comparisons
############################
compare-discrete:
	@echo "Running compare_swarm_discrete.py..."
	$(RUN) experiments/compare_swarm_discrete.py $(ARGS)

compare-continuous:
	@echo "Running compare_swarm_continuous.py..."
	$(RUN) experiments/compare_swarm_continuous.py $(ARGS)

compare-traditional:
	@echo "Running compare_with_traditional.py..."
	$(RUN) experiments/compare_with_traditional.py $(ARGS)
