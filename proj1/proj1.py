import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MOVES = ["R", "P", "S"]
BEATS = {"R": "P", "P": "S", "S": "R"}
IDX = {m: i for i, m in enumerate(MOVES)}
N_ROUNDS = 1000

TRUE_TM = np.array([
    [0.1, 0.6, 0.3],   # after R -> plays R 10%, P 60%, S 30%
    [0.2, 0.2, 0.6],   # after P -> plays R 20%, P 20%, S 60%
    [0.5, 0.3, 0.2],   # after S -> plays R 50%, P 30%, S 20%
])

def markov_move(prev_move):
    row = TRUE_TM[IDX[prev_move]]
    return random.choices(MOVES, weights=row, k=1)[0]

counts = np.ones((3, 3))

def learned_tm():
    return counts / counts.sum(axis=1, keepdims=True)

def bayesian_move(prev_opponent_move):
    row = learned_tm()[IDX[prev_opponent_move]]
    predicted = MOVES[int(np.argmax(row))]
    return BEATS[predicted]

def update_counts(prev_move, next_move):
    counts[IDX[prev_move]][IDX[next_move]] += 1

def outcome(m1, m2):
    if m1 == m2:
        return 0
    if BEATS[m1] == m2:
        return 1
    return -1

def simulate(n=N_ROUNDS):
    score = 0
    score_history = []
    p1_prev = random.choice(MOVES)

    for _ in range(n):
        p1_move = markov_move(p1_prev)
        p2_move = bayesian_move(p1_prev)
        score += outcome(p1_move, p2_move)
        score_history.append(score)
        update_counts(p1_prev, p1_move)
        p1_prev = p1_move

    return score_history

score_history = simulate()

print("=" * 55)
print("  Rock–Paper–Scissors  |  Bayesian Learning Result")
print("=" * 55)
print(f"  Rounds played : {N_ROUNDS}")
print(f"  Final score   : {score_history[-1]:+d}  (+ = P2 wins, - = P1 wins)")
print()
print("  Learned transition matrix (rows = prev move, cols = next):")
print(f"  {'':6}  {'R':>7}  {'P':>7}  {'S':>7}")
tm = learned_tm()
for i, m in enumerate(MOVES):
    print(f"  {m:>6}   {tm[i,0]:6.3f}   {tm[i,1]:6.3f}   {tm[i,2]:6.3f}")
print()
print("  True transition matrix (for comparison):")
print(f"  {'':6}  {'R':>7}  {'P':>7}  {'S':>7}")
for i, m in enumerate(MOVES):
    print(f"  {m:>6}   {TRUE_TM[i,0]:6.3f}   {TRUE_TM[i,1]:6.3f}   {TRUE_TM[i,2]:6.3f}")
print("=" * 55)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

rounds = np.arange(1, N_ROUNDS + 1)
ax.plot(rounds, score_history, color="#4ade80", linewidth=1.5, alpha=0.9)
ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.5)
ax.fill_between(rounds, score_history, 0,
                where=[s > 0 for s in score_history],
                color="#4ade80", alpha=0.12)
ax.fill_between(rounds, score_history, 0,
                where=[s < 0 for s in score_history],
                color="#f87171", alpha=0.12)

ax.set_title("RPS – Bayesian Learner vs Markov Player", color="white", fontsize=14, pad=14)
ax.set_xlabel("Round", color="#94a3b8")
ax.set_ylabel("Accumulated Score  (+ = P2 ahead)", color="#94a3b8")
ax.tick_params(colors="#94a3b8")
for spine in ax.spines.values():
    spine.set_edgecolor("#334155")

ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("rps_score.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
