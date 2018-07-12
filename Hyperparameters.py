
import numpy as np

DAYS = 5
DAILY_CAP = 5
N_ACTIONS = DAYS
STATES = DAYS + 1
ACTIONS = [x for x in range(N_ACTIONS)]
N_STATES = (DAILY_CAP+1)**DAYS

N_EPISODES = 1000
MIN_ALPHA = .00004
ALPHA = MIN_ALPHA
DEMAND_DIST = [1]
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
MAX_EPISODE_STEPS = int(DAYS * DAILY_CAP / np.max(np.array(DEMAND_DIST)))
GAMMA = 0.9
EPS = 1
BATCH_SIZE = 30

MEMORY_SIZE = 50000

PRE_TRAIN_LENGTH = BATCH_SIZE
TRAIN = True

WRITE_RESULTS = False
DESCRIP = "Q-Learning"
MODEL_TYPE = "LINEAR"

env_fname = "./results/" + DESCRIP + \
            "Cap_" + str(DAILY_CAP) + "_" + \
            "Days_" + str(DAYS) + \
            "_" + MODEL_TYPE + "_env.csv"

training_fname = "./results/" + DESCRIP + "_" + \
                 "Days_" + str(DAYS) +\
                 "Cap_" + str(DAILY_CAP) + "_" +\
                 "_" + MODEL_TYPE + "_rewards.csv"

