# Convnetjs-Extras

An implementation of additional functionalities for Convnetjs, including various activation functions.

- LeakyRELU
- ELU
- FReLU
- Swish
- PLU
- PiLU
- DoubleReLU
- Mish
- Softplus

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

### Softplus
Softplus is a smooth and continuous activation function defined as the logarithm of the exponential of the input plus one. It has the advantage of being differentiable everywhere, which allows for stable gradients during training.

## TODO

Some activation functions/loss functions that can be added in the future:

- Gish
- Smish
- Logish
- GeLU
- PReLU
- RReLU
- Softmin
- Softsign
- Softshrink
- Hardshrink
- LogSoftmax





