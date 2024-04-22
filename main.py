import time

import torch

from dqn import DqnAgent, Memory, Transition
from env import BangEnv
from model import SakikoNetwork
from state import StateDevice

BATCH_SIZE = 32


def main():
    qnet = SakikoNetwork(113, 299)
    tnet = SakikoNetwork(113, 299)
    agent = DqnAgent(qnet=qnet, tnet=tnet,
                     rand_func=lambda: torch.rand(22),
                     lr=1e-3, gamma=0.5, epsilon=0.01, epsilon_decay=300)
    memory = Memory(10000)

    device = StateDevice()
    device.start(threaded=True)
    instance = BangEnv(device)

    ep = 0
    mean_rewards = 0
    rewards = 0
    state, _ = instance.reset()
    while state is None:
        time.sleep(0.5)
        state, _ = instance.reset()
    while True:
        action = agent.act_ex(state)
        next_state, reward, done, _ = instance.step(action)
        rewards += reward
        if done:
            device.slide_clear()
            inp = input('>')
            if inp == 'y':
                break
            ep += 1
            mean_rewards = 0.9 * mean_rewards + 0.1 * rewards
            print(f'Episode {ep}, Mean Rewards: {mean_rewards}')
            rewards = 0
            state, _ = instance.reset()
        else:
            memory.push(Transition(state, action, next_state, reward))
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                agent.learn(Transition(*zip(*transitions)))
            state = next_state


if __name__ == '__main__':
    main()
