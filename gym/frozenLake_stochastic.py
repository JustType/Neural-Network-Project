
import gym
import torch
import matplotlib.pyplot as plt
from gym.envs.registration import register

import time



env = gym.make('FrozenLake-v0')

n_episodes = 1000
steps_total = []
rewards_total = []


number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.9
learning_rate = 0.9
egreedy = 0.1

Q = torch.zeros(number_of_states, number_of_actions)


for i_episode in range(n_episodes):

    state = env.reset()
    step = 0

    while True:

        step += 1

        random_for_egreedy = torch.rand(1).item()


        if random_for_egreedy > egreedy:
            random_values = Q[state] + torch.randn(1,number_of_actions) / 1000
            action = torch.max(random_values, 1)[1][0].item()
        else:
            action = env.action_space.sample()
            print 'Random action!'



        new_state, reward, done, info = env.step(action)

        #print new_state
        #print info

        Q[state, action] = (1 - learning_rate) * Q[state,action] + learning_rate * (reward + gamma * torch.max(Q[new_state]))

        state = new_state

        #time.sleep(0.4)

        #env.render()

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            print 'Episode finished after %i steps' % step
            break


print Q

print 'Percent of episodes done successfully {}'.format(sum(rewards_total)/n_episodes)
print 'Percent of episodes done last 100 {}'.format(sum(rewards_total[-100:])/100)
print 'Average number of steps: %.2f' % (sum(steps_total)/n_episodes)
print 'Average number of steps last 100: %.2f' % (sum(steps_total[-100:])/100)

plt.plot(steps_total)
plt.show()
