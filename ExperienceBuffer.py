#https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

import numpy as np
from collections import deque

#an experience is a list of [state, action, reward, next_state, done]
class ExperienceBuffer:
    def __init__(self, buffer_size=5000):
        self.experience_len = 5
        self.buffer = deque(maxlen = buffer_size )

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):

        index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        sample_array = np.array([self.buffer[i] for i in index]).reshape(batch_size, self.experience_len)

        return sample_array

    def empty(self):
        self.buffer = []

    def __str__(self):
        buffer_string = ""
        for buf in self.buffer:
            buffer_string += str(buf) + '\n'



'''
replay = ExperienceBuffer(buffer_size=3)
for i in range(4):
    print(replay.buffer)
    replay.add([[i, 1, -1, 1]])

print(replay.buffer)
foo = replay.sample(2)
print(foo.shape)
'''

