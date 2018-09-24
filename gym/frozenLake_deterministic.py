import gym
import torch
import matplotlib.pyplot as plt
from gym.envs.registration import register

import time


register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)



env = gym.make('FrozenLakeNotSlippery-v0')

n_episodes = 1000
steps_total = []
rewards_total = []
egreedy_total = []


number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
gamma = 0.9
egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999

Q = torch.zeros([number_of_states, number_of_actions])


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


        if egreedy > egreedy_final:
            egreedy *= egreedy_decay


        new_state, reward, done, info = env.step(action)

        #print new_state
        #print info

        Q[state, action] = reward + gamma * torch.max(Q[new_state])

        state = new_state

        #time.sleep(0.4)

        #env.render()

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            egreedy_total.append(egreedy)
            print 'Episode finished after %i steps' % step
            break


print Q

print 'Percent of episodes done successfully {}'.format(sum(rewards_total)/n_episodes)
print 'Percent of episodes done last 100 {}'.format(sum(rewards_total[-100:])/100)
print 'Average number of steps: %.2f' % (sum(steps_total)/n_episodes)
print 'Average number of steps last 100: %.2f' % (sum(steps_total[-100:])/100)

plt.figure(figsize=[12,5])

plt.title('Egreedy decay')
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='green')
plt.show()

plt.figure(figsize=[12,5])
plt.title('Steps')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')
plt.show()
