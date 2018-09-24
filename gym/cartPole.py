## BALANCING POLE ON A CART ###

import gym
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import time


env = gym.make('CartPole-v0')

n_episodes = 500

steps_total = []
rewards_total = []
frames_total = 0
egreedy_total = []

#PARAMS#
gamma = 1
egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 0.98

replay_mem_size = 5000
batch_size = 32

update_target_frequency = 50
clip_error = False
double_dqn = True


##################
seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
##################

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
score_to_solve = 195
solved = False


class NN(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 64, output_size = 1):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, hidden_size)
        self.advantage2 = nn.Linear(hidden_size, output_size)

        self.activate = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)



    def forward(self, x):
        out = self.l(x)
        out = self.activate(out)

        advant = self.advantage(out)
        advant = self.activate(advant)
        advant = self.advantage2(advant)

        value = self.value(out)
        value = self.activate(value)
        value = self.value2(value)

        final = value + advant - advant.mean()



        return final



class QNet(object):
    def __init__(self):
        self.model = NN(input_size=num_states, output_size=num_actions)
        self.target_model = NN(input_size=num_states, output_size=num_actions)
        self.crit = nn.SmoothL1Loss()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.update_target_counter = 0



    def select_action(self, state, epsilon):

        random_for_egreedy = torch.rand(1).item()

        if random_for_egreedy > epsilon:

            with torch.no_grad():
                state = torch.Tensor(state)
                nn_action = self.model(state)
                action = torch.max(nn_action, 0)[1][0].item()

        else:
            action = env.action_space.sample()


        return action


    def optimize(self):

        if len(memory) < batch_size:
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = torch.Tensor(state)
        new_state = torch.Tensor(new_state)
        reward = torch.Tensor(reward)
        action = torch.Tensor(action).long()
        done = torch.Tensor(done)


        if double_dqn:
            new_state_indexes = self.model(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
            new_state_values = self.target_model(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)

        else:
            new_state_values = self.target_model(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]



        target_value = reward + (1 - done) * gamma * max_new_state_values


        predicted_value = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.crit(predicted_value, target_value)
        self.opt.zero_grad()
        loss.backward()


        if clip_error:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

        self.opt.step()

        if self.update_target_counter % update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.update_target_counter += 1



class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity



    def sample(self, batch_size):

        return zip(*random.sample(self.memory, batch_size))


    def __len__(self):
        return len(self.memory)




memory = ExperienceReplay(replay_mem_size)


agent = QNet()

start_time = time.time()

for i_episode in range(n_episodes):

    state = env.reset()
    step = 0
    if egreedy > egreedy_final:
            egreedy *= egreedy_decay

    while True:

        step += 1
        frames_total += 1




        action = agent.select_action(state, egreedy)

        #print action
        #time.sleep(0.5)

        new_state, reward, done, info = env.step(action)

        memory.push(state, action, new_state, reward, done)

        agent.optimize()


        state = new_state
        if i_episode > 2998:
            env._max_episode_steps = 10000
            env.render()
        else:
            env._max_episode_steps = 200


        if done:
            print step
            rewards_total.append(reward)
            steps_total.append(step)
            egreedy_total.append(egreedy)
            report_interval = 10
            elapsed_time = time.time() - start_time
            mean_reward_100 = sum(steps_total[-100:])/100

            if mean_reward_100 > score_to_solve and solved == False:
                print 'Solved!! after {} episodes'.format(i_episode)
                solved = True

            if i_episode % report_interval == 0:
                print '\n***** Episode {} ******' \
                      '\nAverage reward (last {}): {}, last 100: {}, all: {}' \
                      '\nEpsilon: {}, Frames Total: {}' \
                      '\nElapsed time: {}'.format(i_episode,
                                report_interval,
                                sum(steps_total[-report_interval:])/report_interval,
                                sum(steps_total[-100:])/100,
                                sum(steps_total)/len(steps_total),
                                egreedy,
                                frames_total,
                                time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))



            break




#print 'Average reward {}'.format(sum(steps_total)/n_episodes)
print 'Average reward: %.2f' % (sum(steps_total)/n_episodes)
print 'Average reward last 100: %.2f' % (sum(steps_total[-100:])/100)

plt.figure(figsize=[12,5])

plt.title('Rewards')
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green')
plt.show()

plt.figure(figsize=[12,5])
plt.title('Steps')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')
plt.show()

#plt.figure(figsize=[12,5])
#plt.title('Egreedy decay')
#plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue')
#plt.show()


env.close()
env.env.close()
