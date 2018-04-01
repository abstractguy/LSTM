# Counting RNN
import copy, numpy as np
inputs  = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
answers = np.array([[[0, 1]], [[1, 0]], [[1, 1]], [[0, 0]]])
input_synapses  = 2 * np.random.random((2,  16)) - 1
hidden_synapses = 2 * np.random.random((16, 16)) - 1
output_synapses = 2 * np.random.random((16, 2))  - 1
learning_rate, outputs = 0.1, np.zeros_like(answers)
for epoch in range(10000):
	hidden_layers, output_layers, output_errors = [np.zeros(16)], [], []
	input_synapses_update  = np.zeros_like(input_synapses)
	hidden_synapses_update = np.zeros_like(hidden_synapses)
	output_synapses_update = np.zeros_like(output_synapses)
	for time in range(4):
		hidden_layer = 1 / (1 + np.exp(-(np.dot(inputs[time], input_synapses) + np.dot(hidden_layers[-1], hidden_synapses))))
		output_layer = 1 / (1 + np.exp(-(np.dot(hidden_layer, output_synapses))))
		output_errors.append((answers[time] - output_layer) * (output_layer * (1 - output_layer)))
		hidden_layers.append(copy.deepcopy(hidden_layer))
		output_layers.append(copy.deepcopy(output_layer))
	next_hidden_error = np.zeros(16)
	for time in range(4):
		hidden_error = (next_hidden_error.dot(hidden_synapses.T) + output_errors[-time - 1].dot(output_synapses.T)) \
				* (hidden_layers[-time - 1] * (1 - hidden_layers[-time - 1]))
		output_synapses_update += np.atleast_2d(hidden_layers[-time - 1]).T.dot(output_errors[-time - 1])
		hidden_synapses_update += np.atleast_2d(hidden_layers[-time - 2]).T.dot(hidden_error)
		input_synapses_update  += inputs[-time - 1].T.dot(hidden_error)
		next_hidden_error = hidden_error
	input_synapses  += input_synapses_update  * learning_rate
	hidden_synapses += hidden_synapses_update * learning_rate
	output_synapses += output_synapses_update * learning_rate


