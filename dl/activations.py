import math
import torch
import torch.nn as nn


class Activation(nn.Module):
    """
    @brief Base class of activation modules.
    """

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}


class Sigmoid(Activation):
    """
    @brief Sigmoid module
    """

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


class Tanh(Activation):
    """
    @brief Tanh activation module
    """

    def forward(self, x):
        exp_x, exp_neg_x = torch.exp(x), torch.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


class ReLU(Activation):
    """
    @brief ReLU activation module
    """

    def forward(self, x):
        return x * (x > 0).float()


class LeakyReLU(Activation):
    """
    @brief LeakyReLU activation module
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.config = {"alpha": alpha}

    def forward(self, x):
        return torch.where(x > 0, x, x * self.config["alpha"])


class ELU(Activation):
    """
    @brief ELU activation module
    """

    def forward(self, x):
        return torch.where(x > 0, x, torch.exp(x) - 1)


class Swish(Activation):
    """
    @brief Swish activation module
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_grads(act_fn, x):
    """
    Computes the gradients of an activation function at specified positions.
    Inputs:
        act_fn - An object of the class "Activation" with an implemented forward pass.
        x - 1D input tensor.
    Output:
        A tensor with the same size of x containing the gradients of act_fn at x.
    """
    x = x.clone().requires_grad_()
    out = act_fn(x)
    # Summing results in an equal gradient flow for each element of x
    out.sum().backward()
    return x.grad


def vis_act_fn(act_fn, ax, x):
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    x, y, y_grads = x.numpy(), y.numpy(), y_grads.numpy()

    ax.plot(x, y, linewidth=2, label="Value")
    ax.plot(x, y_grads, linewidth=2, label="Grad")
    ax.set_title(act_fn.name)
    ax.set_ylim(-1.5, 2)
    ax.legend()
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(0.5)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")


###################################################
import matplotlib.pyplot as plt

act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
}

print(">> Need to plot activations and the derivaties: [Y/N]")
console_state = input()
if console_state == "Y" or "y":
    # Add activation functions if wanted
    act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
    # Range on which we want to visualize the activation functions
    x = torch.linspace(-5, 5, 1000)
    ## Plotting
    cols = 3
    rows = math.ceil(len(act_fns) / cols)
    fig, ax = plt.subplots(rows, cols)
    for i, act_fn in enumerate(act_fns):
        vis_act_fn(act_fn, ax[divmod(i, cols)], x)
    fig.subplots_adjust(hspace=0.3)
    plt.show()
    plt.close()
