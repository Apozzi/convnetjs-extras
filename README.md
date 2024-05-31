# Convnetjs-Extras

An implementation of additional functionalities for Convnetjs, including various activation functions.

- LeakyRELU
- ELU
- FReLU
- Swish
- PLU

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

## TODO

Some activation functions/loss functions that can be added in the future:

- PiLU
- DoubleRELU
- Mish
- Logish
- Softplus
- GeLU
- PReLU
- Softmin
- Softsign
- Softshrink
- Hardshrink





