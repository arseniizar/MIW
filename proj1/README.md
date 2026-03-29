# Project 1 — Rock-Paper-Scissors with Bayesian Learning

Simulation of RPS where a Markov player follows a fixed transition matrix and a Bayesian learner adapts to beat it over 1000 rounds.

## Usage

```bash
python proj1.py
```

Outputs the learned vs true transition matrix to stdout and saves `rps_score.png`.

## How it works

Player 1 samples moves from a fixed 3×3 transition matrix (Markov chain). Player 2 maintains a count matrix of observed transitions, normalizes it into a probability estimate, and always plays the counter-move to the most probable next move — equivalent to Bayesian updating with a uniform prior. After 1000 rounds the learned matrix converges toward the true one.