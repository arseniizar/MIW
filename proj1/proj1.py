import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MOVES = ["R", "P", "S"]
BEATS = {"R": "P", "P": "S", "S": "R"}
IDX   = {m: i for i, m in enumerate(MOVES)}

N_ROUNDS = 1000

TRUE_TM = np.array([
    [0.1, 0.6, 0.3],
    [0.2, 0.2, 0.6],
    [0.5, 0.3, 0.2],
])

counts = np.ones((3, 3))


def markov_move(prev):
    return random.choices(MOVES, weights=TRUE_TM[IDX[prev]], k=1)[0]


def learned_tm():
    return counts / counts.sum(axis=1, keepdims=True)


def counter_move(prev):
    predicted = MOVES[int(np.argmax(learned_tm()[IDX[prev]]))]
    return BEATS[predicted]


def update(prev, nxt):
    counts[IDX[prev]][IDX[nxt]] += 1


def outcome(m1, m2):
    if m1 == m2:
        return 0
    return 1 if BEATS[m1] == m2 else -1


def simulate(n=N_ROUNDS):
    score, history = 0, []
    prev = random.choice(MOVES)
    for _ in range(n):
        p1 = markov_move(prev)
        p2 = counter_move(prev)
        score += outcome(p1, p2)
        history.append(score)
        update(prev, p1)
        prev = p1
    return history


def print_results(history):
    tm = learned_tm()
    print("=" * 55)
    print("  Rock-Paper-Scissors  |  Bayesian Learning Result")
    print("=" * 55)
    print(f"  Rounds played : {N_ROUNDS}")
    print(f"  Final score   : {history[-1]:+d}  (+ = P2 wins, - = P1 wins)")
    print()
    print("  Learned transition matrix:")
    print(f"  {'':6}  {'R':>7}  {'P':>7}  {'S':>7}")
    for i, m in enumerate(MOVES):
        print(f"  {m:>6}   {tm[i,0]:6.3f}   {tm[i,1]:6.3f}   {tm[i,2]:6.3f}")
    print()
    print("  True transition matrix:")
    print(f"  {'':6}  {'R':>7}  {'P':>7}  {'S':>7}")
    for i, m in enumerate(MOVES):
        print(f"  {m:>6}   {TRUE_TM[i,0]:6.3f}   {TRUE_TM[i,1]:6.3f}   {TRUE_TM[i,2]:6.3f}")
    print("=" * 55)


def plot(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    rounds = np.arange(1, N_ROUNDS + 1)
    ax.plot(rounds, history, color="#4ade80", linewidth=1.5, alpha=0.9)
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(rounds, history, 0,
                    where=[s > 0 for s in history], color="#4ade80", alpha=0.12)
    ax.fill_between(rounds, history, 0,
                    where=[s < 0 for s in history], color="#f87171", alpha=0.12)

    ax.set_title("RPS - Bayesian Learner vs Markov Player", color="white", fontsize=14, pad=14)
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


if __name__ == "__main__":
    history = simulate()
    print_results(history)
    plot(history)
