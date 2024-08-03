import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the input values and weights based on your code
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([2.0, 3.0, 0.5])

# Calculate outputs for the first input sample
sample_index = 0  # Index of the sample to visualize
outputs = np.dot(inputs[sample_index], weights.T) + biases

# Create a graph
G = nx.DiGraph()

# Add nodes for inputs
for i in range(inputs.shape[1]):
    G.add_node(f'Input {i+1}', pos=(0, -i*3.5), value=inputs[sample_index, i])

# Add nodes for hidden layer neurons
for j in range(weights.shape[0]):
    G.add_node(f'Neuron {j+1}', pos=(3, -j*3.5), bias=biases[j])

# Add nodes for outputs
for k in range(weights.shape[0]):
    G.add_node(f'Output {k+1}', pos=(6, -k*3.5), value=outputs[k])

# Add edges with weights from inputs to hidden layer neurons
for i in range(inputs.shape[1]):
    for j in range(weights.shape[0]):
        G.add_edge(f'Input {i+1}', f'Neuron {j+1}', weight=weights[j, i])

# Add edges from hidden neurons to outputs without weights
for j in range(weights.shape[0]):
    G.add_edge(f'Neuron {j+1}', f'Output {j+1}')  # Omitting weight labels

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Define node groups for coloring
input_nodes = [f'Input {i+1}' for i in range(inputs.shape[1])]
hidden_nodes = [f'Neuron {j+1}' for j in range(weights.shape[0])]
output_nodes = [f'Output {k+1}' for k in range(weights.shape[0])]

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='skyblue', node_size=1500)
nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightgreen', node_size=1500)
nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='lightcoral', node_size=1500)

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges, arrowstyle='-|>', arrowsize=20)

# Add labels for nodes, including neuron labels in the hidden layer
node_labels = {n: f"{n}" for n in G.nodes}
node_labels.update({f'Output {k+1}': f"Output {k+1}\n{outputs[k]:.2f}" for k in range(weights.shape[0])})
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

# Add biases under each hidden neuron
for j in range(weights.shape[0]):
    bias_label = f"Bias: {biases[j]:.2f}"
    x, y = pos[f'Neuron {j+1}']
    plt.text(x, y - 1.5, bias_label, fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

# Add labels for edges (weights) with adjusted positions and spacing
edge_labels = nx.get_edge_attributes(G, 'weight')

# Position labels differently for each edge to prevent overlapping
for (start, end, weight) in G.edges(data='weight'):
    if weight is not None:
        label_pos = 0.2 if "Input 1" in start else 0.5 if "Input 2" in start else 0.8
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(start, end): f"{weight:.2f}"},
            font_size=8, label_pos=label_pos,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

# Add a text box for the input sample at the bottom right
sample_text = f"Input Sample: {inputs[sample_index]}"
plt.gcf().text(0.95, 0.15, sample_text, fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Add a text box for weights connected to each input below the input sample box
weights_text = "\n".join([f"Input {i+1} Weights: {weights[:, i]}" for i in range(weights.shape[1])])
plt.gcf().text(0.95, 0.05, weights_text, fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.title("""Neural Network Visualization with Weights and Biases
          >> 4 Inputs into 3 Neurons <<""")
plt.axis('off')
plt.show()
