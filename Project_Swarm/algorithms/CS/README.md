Project: AI-ProJ1



This project contains implementations of several metaheuristic optimizers and an experiment runner to compare them on benchmark functions (Rastrigin).



Structure:



\- `optimizers/` : optimizer implementations (Hill Climbing, Simulated Annealing, Genetic Algorithm, plus `cuckoo\_search.py` at project root)

\- `benchmarks/` : benchmark functions (Rastrigin) and wrappers that count evaluations

\- `experiments/runner.py` : simple runner to execute multiple runs and plot convergence

\- `tools/plotting.py` : helper plotting functions



Quick start (PowerShell):



```powershell

python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python experiments\\runner.py

```



