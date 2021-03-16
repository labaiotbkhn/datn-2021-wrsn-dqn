import numpy as np


def updateMemories(q_learning, deep_qlearning):
    state_id = q_learning.state
    reward = q_learning.reward[state_id]
    input_state_dqn = np.reshape(q_learning.input_state_dqn, [
                                 1, deep_qlearning.state_size])

    deep_qlearning.memorize(input_state_dqn, state_id, reward, input_state_dqn)


def updateNextAction(deep_qlearning, next_state):
    if len(deep_qlearning.memory) > 0:
        last_memory = list(deep_qlearning.memory[-1])
        last_memory[3] = next_state
        deep_qlearning.memory[-1] = tuple(last_memory)


def _build_input_state(network):
    list_state = []
    list_state.append(network.mc.current[0])
    list_state.append(network.mc.current[1])
    list_state.append(network.mc.energy)
    for nd in network.node:
        list_state.append(nd.energy)
        list_state.append(nd.avg_energy)

    return np.array(list_state)
