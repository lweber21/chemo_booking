import BookEnv
import numpy as np
import random
import matplotlib.pyplot as plt
import Hyperparameters as param
import os
import csv
#references
#https://medium.com/@curiousily/solving-an-mdp-with-q-learning-from-scratch-deep-reinforcement-learning-for-hackers-part-1-45d1d360c120
#https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe

#MODEL_TYPE = "TABLE"

#define hyperparameters
# DAYS = 5
# DAILY_CAP = 10
# N_ACTIONS = DAYS
# ACTIONS = [x for x in range(N_ACTIONS)]
# N_STATES = (DAILY_CAP+1)**DAYS
#
# N_EPISODES = 10000
# MIN_ALPHA = 1e-7
# DEMAND_DIST = [1, 2, 3]
# alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
# MAX_EPISODE_STEPS = int(DAYS * DAILY_CAP / np.max(np.array(DEMAND_DIST)))
# GAMMA = 0.90
# EPS = 0.1

q_table = {}

eps = param.EPS

def q(state, action =None):

    #convert state to tuple so it is hashable
    state_tuple = tuple(state.reshape(1, -1)[0])

    if state_tuple not in q_table:
        #q_table.update({state: np.zeros(len(ACTIONS))})
        q_table[state_tuple] = np.zeros(len(param.ACTIONS))

    if action is None:
        return q_table[state_tuple]

    return q_table[state_tuple][action]


def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(param.ACTIONS)
    else:
        return np.argmax(q(state))


env = BookEnv.BookingEnv(days=param.DAYS, daily_avail=param.DAILY_CAP, demand_dist=param.DEMAND_DIST)

rList = []
eList = []
dList = []
for e in range(param.N_EPISODES):
    total_reward = 0
    env.reset()
    state = env.state
    patient = env.Patient

    alpha = param.alphas[e]

    patient_id = 1

    for _ in range(param.MAX_EPISODE_STEPS):

        action = choose_action(state)
        next_state, next_patient, reward, done = env.step(action, patient)
        total_reward += reward
        q(state)[action] = q(state, action) + alpha * (reward + param.GAMMA * np.max(q(next_state)) - q(state, action))

        state = next_state
        patient = next_patient
        patient_id += 1
        if done:
            break
        eps = 1. / ((e / 50) + 10)

    print(f"Episode {e + 1}: total reward -> {total_reward}")
    rList.append(total_reward)
    eList.append(e)
    dList.append(env.total_demand)

plt.plot(eList, rList)
plt.show()

if param.WRITE_RESULTS:
    with open(param.training_fname, 'w', newline='') as f:
        thewriter = csv.writer(f)

        thewriter.writerow(["epoch", "total_reward", "total_demand"])
        for e in eList:
            thewriter.writerow([e + 1, rList[e], dList[e]])

#lets put our agent to work
env.reset()
state = env.state
patient = env.Patient
total_rewards = 0
j = 0
rList = []

while j <= param.MAX_EPISODE_STEPS:
    j += 1
    action = choose_action(state)
    next_state, next_patient, reward, done = env.step(action, patient)
    total_rewards += reward
    state = next_state
    patient = next_patient

    rList.append(total_rewards)
print(f"total reward -> {total_rewards}")
print(env)
path = "./results"

if param.WRITE_RESULTS:

    env.write_env(param.env_fname)
