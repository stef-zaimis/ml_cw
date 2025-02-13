import numpy as np
from hmmlearn import hmm

GRID_HEIGHT=3
GRID_WIDTH=3

with open("/home/stefanos/uni/ml/cw/rewards.txt", "r") as file:
    rewards = np.array([int(line.strip()) for line in file])

n_states = GRID_HEIGHT*GRID_WIDTH
n_observations = 3

model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=1e-4, random_state=42)

trans_prob = np.zeros((n_states, n_states))

for i in range(GRID_HEIGHT):
    for j in range(GRID_WIDTH):
        neighbours = []
        if i>0: neighbours.append((i-1, j))
        if i<2: neighbours.append((i+1, j))
        if j>0: neighbours.append((i,j-1))
        if j<2: neighbours.append((i,j+1))

        idx = i*GRID_WIDTH+j
        prob = 1/len(neighbours)
        for ni, nj in neighbours:
            neighbour_idx = ni*GRID_WIDTH + nj
            trans_prob[idx, neighbour_idx] = prob

start_prob = np.full(n_states, 1/n_states)
emit_prob = np.random.rand(n_states, n_observations)
emit_prob /= emit_prob.sum(axis=1, keepdims=True)

model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emissionprob_ = emit_prob

rewards = rewards.reshape(-1, 1)
model.fit(rewards)

print(f"Learned Starting Probabilities: {model.startprob_}")
print(f"\nLearned Transition Probabilities: {model.transmat_}")
print(f"\nLearned Emission Probabilities: {model.emissionprob_}")

hidden_states = model.predict(rewards)
print(f"\nPreddicted Hidden States: {hidden_states}")
