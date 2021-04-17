import numpy as np


def updateMemories(q_learning, deep_qlearning):
    next_action = q_learning.state
    reward = q_learning.reward_dqn
    input_state_dqn = np.reshape(q_learning.input_state_dqn, [
                                 1, deep_qlearning.state_size])
    print("reward: {}".format(reward))

    # update temporary memories
    deep_qlearning.memorize(input_state_dqn, next_action,
                            reward, input_state_dqn)


def updateNextAction(deep_qlearning, next_state):
    if len(deep_qlearning.memory) > 0:
        last_memory = list(deep_qlearning.memory[-1])
        last_memory[3] = next_state
        deep_qlearning.memory[-1] = tuple(last_memory)


def _build_input_state(network):
    list_state = []
    # normalize data to pass network
    energies = [nd.energy for nd in network.node]

    avg_energies = [nd.avg_energy for nd in network.node]
    # max_energy = max(energies)
    # min_energy = min(energies)
    # max_avg = max(avg_energies)
    # min_avg = min(avg_energies)

    # list_energy_normalize = [(nd.energy / max_energy )  for nd in network.node]
    # list_avg_energy_normalize = [(nd.avg_energy / max_avg) for nd in network.node]

    list_state.extend(energies)
    list_state.extend(avg_energies)
    return np.array(list_state)
