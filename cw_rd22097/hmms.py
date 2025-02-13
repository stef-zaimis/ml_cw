import numpy as np
from hmmlearn import hmm

rewards = np.loadtxt('rewards.txt').astype(int).reshape(-1, 1)

n_states = 9 

# Code Task 14:
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01, init_params="ste", params="ste", verbose=True)

model.fit(rewards)

print("Learned Initial State Probabilities (startprob_):\n", model.startprob_)
print("\nLearned Transition Matrix (transmat_):\n", model.transmat_)
print("\nLearned Emission Probabilities (emissionprob_):\n", model.emissionprob_)

# Code Task 15:
def neighbors(state):
    x, y = divmod(state, 3)
    moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [nx * 3 + ny for nx, ny in moves if 0 <= nx < 3 and 0 <= ny < 3]

true_transmat = np.zeros((n_states, n_states))
for state in range(n_states):
    nbrs = neighbors(state)
    prob = 1 / len(nbrs) if nbrs else 0
    for nbr in nbrs:
        true_transmat[state, nbr] = prob

model_true_transmat = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01, init_params="se", params="se", verbose=True)
model_true_transmat.transmat_ = true_transmat

model_true_transmat.fit(rewards)

print("\nLearned Initial State Probabilities (startprob_):\n", model_true_transmat.startprob_)
print("\nFixed Transition Matrix (transmat_):\n", model_true_transmat.transmat_)
print("\nLearned Emission Probabilities (emissionprob_):\n", model_true_transmat.emissionprob_)

