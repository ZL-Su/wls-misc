# Pose representation of dymanic camera

- Rotation: $\dot{R}(t) = \hat{w}R(t)$ -> 1st-order approximation $$\dot{R}(t) = I + \hat{w}(0)dt$$
- Rigid motion: $\dot{T}(t) = \hat{\xi}T(t)$, where $\hat{\xi} \in se(3)$ is the twist and $\xi\in\mathbb{R}^6$ the twist coordinates.

<span style="color:green">**Lie Group and Lie Algebra**</span> - $SO(3)$ and $\mathfrak{so}(3)$ for rotation and $SE(3)$ and $\mathfrak{se}(3)$ for rigid transformation

1. Lie group is a smooth manifold
2. Lie algebra is the tangent space of the Lie group at identity location

# Convolution

$\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)$

## Output size of convolution

- Height $H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]\times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor$
- Width $W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor$

## Neuron operation

- $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$
