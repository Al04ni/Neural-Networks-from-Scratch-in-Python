import matplotlib.pyplot as plt
import networkx as nx

# Define the number of neurons in each layer
n_inputs = 2
n_neurons_layer1 = 3

# Feature names
features = ["Feature 1", "Feature 2"]

# Create a graph object
G = nx.DiGraph()

# Add nodes for input layer
for i in range(n_inputs):
    G.add_node(f"Input ({features[i]})", layer=0)

# Add nodes for dense layer 1
for j in range(n_neurons_layer1):
    G.add_node(f"Dense1 Neuron {j+1}", layer=1)

# Add edges (connections) from input layer to dense layer 1
for i in range(n_inputs):
    for j in range(n_neurons_layer1):
        G.add_edge(f"Input ({features[i]})", f"Dense1 Neuron {j+1}")

# Define node positions
pos = {}
layer_sizes = [n_inputs, n_neurons_layer1]

for layer, size in enumerate(layer_sizes):
    for i in range(size):
        pos[f"{['Input', 'Dense1 Neuron'][layer]} {i+1}"] = (layer, size - i - 1)

# Update positions for input layer with feature names
for i, feature in enumerate(features):
    pos[f"Input ({feature})"] = (0, n_inputs - i - 1)

# Plot the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)
plt.title("Neural Network Structure with Feature Names")
plt.show()