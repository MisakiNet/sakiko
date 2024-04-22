import torch
from tqdm import trange

from dqn import DqnAgent, Memory, Transition
from env import BangEnv
from model import SakikoNetwork
from state import StateDevice

BATCH_SIZE = 32

device = torch.device('cuda')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_device(device)

qnet = SakikoNetwork(113, 299)
tnet = SakikoNetwork(113, 299)

device = StateDevice()
instance = BangEnv(device)
device.start(threaded=True)


def train_from(q_path, memory_path):
    torch.load(q_path, qnet)
    memory = Memory.load(memory_path, 10000)

    agent = DqnAgent(qnet=qnet, tnet=tnet,
                     rand_func=lambda: torch.rand(22),
                     lr=1e-3, gamma=0.5, epsilon=0.01, epsilon_decay=0,
                     target_update_batch=5)

    ep = 0
    rewards = 0
    state, _ = instance.reset()
    while True:
        action = agent.act_ex(state)
        next_state, reward, done = instance.step(action)
        rewards = rewards * 0.9 + reward * 0.1
        if done:
            device.slide_clear()
            ep += 1
            print(f'Episode {ep}, Mean Rewards: {rewards}')
            eps = int(input('>'))
            if len(memory) > BATCH_SIZE:
                pbar = trange(eps)
                for _ in pbar:
                    transitions = memory.sample(BATCH_SIZE)
                    loss = agent.learn(Transition(*zip(*transitions)))
                    pbar.set_description(f'Loss: {loss}')
            rewards = 0
            state, _ = instance.reset()
        else:
            memory.push(Transition(state, action, reward, next_state))
            state = next_state


def evaluate(q_path):
    qnet.load_state_dict(torch.load(q_path))
    agent = DqnAgent(qnet=qnet, tnet=tnet,
                     rand_func=lambda: torch.rand(22),
                     lr=1e-3, gamma=0.5, epsilon=0.01, epsilon_decay=0,
                     target_update_batch=5)
    rewards = 0
    input('Press Enter to start: ')
    print('Ready!')
    state, _ = instance.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = instance.step(action)
        rewards = rewards * 0.9 + reward * 0.1
        state = next_state
    print(f'Rewards: {rewards}')


def main():
    agent = DqnAgent(qnet=qnet, tnet=tnet,
                     rand_func=lambda: torch.rand(22),
                     lr=1e-3, gamma=0.5, epsilon=0.01, epsilon_decay=300,
                     target_update_batch=5)
    memory = Memory(10000)

    ep = 0
    mean_rewards = 0
    rewards = 0
    state, _ = instance.reset()
    while True:
        action = agent.act_ex(state)
        next_state, reward, done = instance.step(action)
        rewards += reward
        if done:
            device.clear()
            ep += 1
            mean_rewards = 0.9 * mean_rewards + 0.1 * rewards
            print(f'Episode {ep}, Mean Rewards: {mean_rewards}')

            memory_path = input('Save memory (memory/?.pkl): ')
            if memory_path != 'n':
                memory.save(f'memory/{memory_path}.pkl')

            inp = input('>')
            if inp == 'y':
                break
            eps = int(inp)
            if len(memory) > BATCH_SIZE:
                pbar = trange(eps)
                for _ in pbar:
                    transitions = memory.sample(BATCH_SIZE)
                    loss = agent.learn(Transition(*zip(*transitions)))
                    pbar.set_description(f'Loss: {loss}')
            rewards = 0
            state, _ = instance.reset()
        else:
            memory.push(Transition(state, action, reward, next_state))
            state = next_state


if __name__ == '__main__':
    main()
    # train_from('checkpoint/b.pth', 'memory/c.pkl')
    # evaluate('checkpoint/b_s.pth')
