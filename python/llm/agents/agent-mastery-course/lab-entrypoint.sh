#!/bin/bash
set -euo pipefail

cat <<'BANNER'

╔══════════════════════════════════════════════════════════════╗
║         AI Trip Planner — Agent Mastery Course              ║
╚══════════════════════════════════════════════════════════════╝

  ✓ Jupyter Lab is ready!

  Open http://localhost:8888 in your browser
  Then open: labs/lab1and2_base_agent.ipynb

  Labs in this course:
    lab1and2_base_agent.ipynb   ← Start here
    lab3_agent_architectures.ipynb
    lab4_tools.ipynb
    lab5_RAG.ipynb
    lab6_evals.ipynb

  To stop: press Ctrl+C in this terminal

BANNER

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --LabApp.default_url=/lab/tree/labs
