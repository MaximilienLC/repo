import random

import torch
from neuroevolution.dynamic_net import Net

torch.manual_seed(0)
random.seed(0)


class WelfordRunningStandardizer:
    def __init__(self, n_mean_M2):
        self.n_mean_M2 = n_mean_M2

    def __call__(self, x):
        self.n_mean_M2[:, 0] += 1
        delta = x - self.n_mean_M2[:, 1]
        self.n_mean_M2[:, 1] += delta / self.n_mean_M2[:, 0]
        delta_new = x - self.n_mean_M2[:, 1]
        self.n_mean_M2[:, 1] += delta * delta_new
        variance = self.n_mean_M2[:, 2] / self.n_mean_M2[:, 0]
        std_dev = torch.sqrt(variance)
        is_valid = self.n_mean_M2[:, 0] >= 2
        safe_std_dev = torch.clamp(std_dev, min=1e-8)
        raw_z_score = (x - self.n_mean_M2[:, 1]) / safe_std_dev
        return torch.where(is_valid, raw_z_score, torch.tensor(0.0))


POPULATION_SIZE = 4
NUM_INPUTS = 3
NUM_OUTPUTS = 2
nets = [Net(NUM_INPUTS, NUM_OUTPUTS) for _ in range(POPULATION_SIZE)]
for net in nets:
    for _ in range(2):  # just to grow the networks a bit
        net.mutate()
nets_num_nodes = torch.tensor([len(net.nodes.all) for net in nets])
print(1)
print(nets_num_nodes)
# We add a value at the front to aid computation. This value at index 0
# will always output 0. Empty in-node slots map to 0, meaning that node.
x = torch.cat(([torch.zeros(1)] + [net.x for net in nets]))  # [0]: zero-returning node
print(2)
print(x)
print(x.shape)
n_mean_M2 = torch.cat(([torch.zeros(1, 3)] + [net.n_mean_M2 for net in nets]))
wrs = WelfordRunningStandardizer(n_mean_M2)
print(3)
print(n_mean_M2)
print(n_mean_M2.shape)
x_input_nodes_start_indices = (
    torch.cat((torch.tensor([0]), torch.cumsum(nets_num_nodes[:-1], dim=0))) + 1
)
print(4)
print(x_input_nodes_start_indices)
x_input_nodes_indices = (
    x_input_nodes_start_indices.unsqueeze(1) + torch.arange(NUM_INPUTS)
).flatten()
print(5)
print(x_input_nodes_indices)
x_mutable_nodes_indices = torch.arange(1, len(x))[
    ~torch.isin(torch.arange(1, len(x)), x_input_nodes_indices)
]
print(6)
print(x_mutable_nodes_indices)
nets_num_mutable_nodes = nets_num_nodes - NUM_INPUTS
nets_cum_num_mutable_nodes = torch.cumsum(nets_num_mutable_nodes, 0)
x_in_nodes_indices = torch.empty((nets_num_mutable_nodes.sum(), 3), dtype=torch.int32)
for i in range(POPULATION_SIZE):
    start = 0 if i == 0 else nets_cum_num_mutable_nodes[i - 1]
    end = nets_cum_num_mutable_nodes[i]
    net_in_nodes_indices = nets[i].in_nodes_indices
    x_in_nodes_indices[start:end] = (
        net_in_nodes_indices
        + (net_in_nodes_indices >= 0) * x_input_nodes_start_indices[i]
    )
x_in_nodes_indices = torch.relu(x_in_nodes_indices)  # Map the -1s to 0s
flat_x_in_nodes_indices = x_in_nodes_indices.flatten()
print(7)
print(x_in_nodes_indices)
print(x_in_nodes_indices.shape)
weights = torch.cat([net.weights for net in nets])
print(8)
print(weights)
print(weights.shape)
# START LOOP
obs = torch.randn(POPULATION_SIZE, NUM_INPUTS)  # from env
print(9)
print(obs)
print(obs.shape)
flat_obs = obs.flatten()
print(10)
print(flat_obs)
print(flat_obs.shape)
x[x_input_nodes_indices] = flat_obs
print(11)
print(x)
print(x.shape)
# From 2nd iter:
# x = wrs(x)
# print(14)
# print(x)
# print(x.shape)
y = torch.gather(x, 0, flat_x_in_nodes_indices).reshape(-1, 3)
print(12)
print(y)
print(y.shape)
z = (y * weights).sum(dim=1)
print(13)
print(z)
print(z.shape)
x[x_mutable_nodes_indices] = z
print(14)
print(x)
print(x.shape)
# END LOOP
for i in range(POPULATION_SIZE):
    start = x_input_nodes_start_indices[i]
    end = None if i + 1 == POPULATION_SIZE else x_input_nodes_start_indices[i + 1]
    nets[i].x = x[start:end]
    nets[i].n_mean_M2 = wrs.n_mean_M2[start:end]
