# Convnetjs-Extras

An implementation of additional functionalities for Convnetjs, including various activation functions.

- LeakyRELU
- ELU
- FReLU
- PLU
- PiLU
- DoubleReLU
- Swish
- Mish
- Gish
- Logish
- Softplus
- Softmin
- Softsign
- Softshrink
- Hardshrink

## How to Use Convnetjs-Extras?

Simply add the convnetjs-extras scripts along with the convnetjs library.
```javascript
<script src="modules/convnet-min.js"></script>
<script src="modules/convnet-extras-min.js"></script>
```

## Activation Functions
### LeakyRELU
Leaky Rectified Linear Unit (LeakyRELU) is an activation function that allows a small, non-zero gradient when the unit is not active, which helps to keep the information flowing through the network during training.

### ELU
Exponential Linear Unit (ELU) is an activation function that tends to converge cost to zero faster and produce more accurate results. It adds a smooth transition to negative inputs, reducing the vanishing gradient problem.

### FReLU
Fixed Rectified Linear Unit (FReLU) is a variant of the ReLU activation function that includes a fixed parameter for the threshold, adding flexibility to the learning process.

### Swish/SiLU
Swish is a smooth, non-monotonic activation function that tends to perform better than ReLU on deeper models. It is defined as the product of the input and its sigmoid function.

### PLU
Piecewise Linear Unit (PLU) is an activation function that allows for multiple linear segments, which can help in capturing complex patterns in the data.

### PiLU
Piecewise Linear Unit (PiLU) is an activation function that introduces two linear segments with different slopes, creating a piecewise linear transition. It enhances the representational capacity of the network by allowing for more flexible and diverse transformations of the input data.

### DoubleReLU
DoubleReLU is an activation function composed of two rectified linear units (ReLU) applied sequentially. It provides a simple yet effective way to introduce non-linearity into the network while maintaining computational efficiency. The double application of ReLU allows for a more pronounced transformation of the input data, potentially capturing more complex patterns in the data.

### Mish
Mish is a self-regularized activation function that smoothly interpolates between the linear and nonlinear regimes. It has shown promising results in various deep learning tasks, often outperforming traditional activation functions like ReLU.

### Gish
Gish is a novel activation function that combines exponential and logarithmic transformations to provide robust non-linearity. It has demonstrated strong performance in deep learning models by effectively handling negative inputs and providing smooth gradients, which can enhance training stability and model performance.

### Softplus
Softplus is a smooth and continuous activation function defined as the logarithm of the exponential of the input plus one. It has the advantage of being differentiable everywhere, which allows for stable gradients during training.

### Logish
Logish is an activation function that blends the properties of the logarithmic function with the sigmoid function. This function is designed to offer a balance between non-linearity and stability, potentially improving convergence and performance in various neural network architectures.

### Softmin
Softmin is an activation function that applies the softmin operation to its inputs, effectively transforming them into a probability distribution where smaller values are amplified. It is particularly useful for tasks where the goal is to emphasize smaller input values.

### Softsign
Softsign is a smooth and differentiable activation function that approximates the sign function with a soft transition. This function provides a continuous approximation of the sign function, helping to mitigate the problem of vanishing gradients and improving the learning dynamics in neural networks.

### Softshrink
Softshrink is a thresholding activation function that introduces sparsity by shrinking values towards zero. This function is useful for regularization and feature selection, as it helps to reduce the impact of small values and promote sparsity in the activations.

### Hardshrink
Hardshrink is a simple threshold-based activation function that sets values within a specific range to zero. This function is effective in scenarios where you want to introduce sparsity and handle outliers by zeroing out values within a certain range.

## TODO

Some activation functions/loss functions that can be added in the future:

- Smish
- GeLU
- PReLU
- RReLU
- LogSoftmax





