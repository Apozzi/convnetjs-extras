# Convnetjs-Extras

An implementation of additional functionalities for Convnetjs, including various activation functions.

- LeakyRELU
- ELU
- FReLU
- GeLU
- PReLU
- RReLU
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

**Definition**: The Leaky Rectified Linear Unit (LeakyRELU) activation function is defined as:

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'leaky_relu', alpha: 0.1}); 
```

where $\alpha$ is a small constant (by default 0.01) that determines the slope of the function for negative inputs.


### ELU
Exponential Linear Unit (ELU) is an activation function that tends to converge cost to zero faster and produce more accurate results. It adds a smooth transition to negative inputs, reducing the vanishing gradient problem.

**Definition**: The Exponential Linear Unit (ELU) activation function is defined as:

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0 
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha e^x & \text{if } x \leq 0 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'elu', alpha: 0.01}); 
```

where $\alpha$ is a hyperparameter that controls the value to which the function asymptotically approaches for negative inputs.


### FReLU
Flexible Rectified Linear Units (FReLU) is a variant of the ReLU activation function that includes a fixed parameter for the threshold, adding flexibility to the learning process.

**Definition**: The Fixed Rectified Linear Unit (FReLU) activation function is defined as:

$$
f(x) = \text{ReLU}(x) + b
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'frelu'}); 
```

### Swish/SiLU
Swish is a smooth, non-monotonic activation function that tends to perform better than ReLU on deeper models. It is defined as the product of the input and its sigmoid function.

**Definition**: The Swish (or Sigmoid Linear Unit, SiLU) activation function is defined as:

$$
f(x) = x \cdot \sigma(x)
$$


**Derivative:**

$$
f'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))
$$

where $\sigma(x)$ is the sigmoid function.

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'swish'}); 
```

### PLU
Piecewise Linear Unit (PLU) is an activation function that allows for multiple linear segments, which can help in capturing complex patterns in the data.

**Definition**: The Piecewise Linear Unit (PLU) activation function is defined as:

$$
f(x) = \max(\alpha (x + c) - c, \min(\alpha (x - c) + c, x))
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
\alpha & \text{if } x < -c \\
1 & \text{if } -c \leq x \leq c \\
\alpha & \text{if } x > c 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'plu'}); 
```

### PiLU
Piecewise Linear Unit (PiLU) is an activation function that introduces two linear segments with different slopes, creating a piecewise linear transition. It enhances the representational capacity of the network by allowing for more flexible and diverse transformations of the input data.

**Definition**: The Piecewise Linear Unit (PiLU) activation function is defined as:

$$
f(x) = \begin{cases} 
\alpha x + \gamma (1 - \alpha), & x > \gamma \\
\beta x + \gamma (1 - \beta), & x \leq \gamma 
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
\alpha & \text{if } x > \gamma \\
\beta & \text{if } x \leq \gamma 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'pilu'}); 
```

### DoubleReLU
DoubleReLU is an activation function composed of two rectified linear units (ReLU) applied sequentially. It provides a simple yet effective way to introduce non-linearity into the network while maintaining computational efficiency. The double application of ReLU allows for a more pronounced transformation of the input data, potentially capturing more complex patterns in the data.

**Definition**: The Double Rectified Linear Unit (DoubleReLU) activation function is defined as:

$$
f(x) = \begin{cases} 
x - \alpha, & x > \alpha \\
0, & -\alpha \leq x \leq \alpha \\
x + \alpha, & x < -\alpha 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'double_relu', alpha:0.5}); 
```

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > \alpha \text{ or } x < -\alpha \\
0 & \text{if } -\alpha \leq x \leq \alpha 
\end{cases}
$$

### Mish
Mish is a self-regularized activation function that smoothly interpolates between the linear and nonlinear regimes. It has shown promising results in various deep learning tasks, often outperforming traditional activation functions like ReLU.

**Definition**: The Mish activation function is defined as:

$$
f(x) = x \cdot \tanh(\text{softplus}(x))
$$

**Derivative:**

$$
f'(x) = \tanh(\text{softplus}(x)) + x \cdot \text{sech}^2(\text{softplus}(x)) \cdot \sigma(x)
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'mish'}); 
```

### Gish
Gish is a novel activation function that combines exponential and logarithmic transformations to provide robust non-linearity. It has demonstrated strong performance in deep learning models by effectively handling negative inputs and providing smooth gradients, which can enhance training stability and model performance.

**Definition**: The Gish activation function is defined as:

$$
f(x) = x \cdot \ln\left(2 - e^{-e^x}\right)
$$

**Derivative:**

$$
f'(x) = \ln\left(2 - e^{-e^x}\right) + x \cdot \frac{e^x \cdot e^{-e^x}}{2 - e^{-e^x}}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'gish'}); 
```

### Softplus
Softplus is a smooth and continuous activation function defined as the logarithm of the exponential of the input plus one. It has the advantage of being differentiable everywhere, which allows for stable gradients during training.

**Definition**: The Softplus activation function is defined as:

$$
f(x) = \ln(1 + \exp(x))
$$

**Derivative:**

$$
f'(x) = \sigma(x)
$$

**Example:**

```javascript
layer_defs.push({type:'softplus', num_classes:10});
```


### Logish
Logish is an activation function that blends the properties of the logarithmic function with the sigmoid function. This function is designed to offer a balance between non-linearity and stability, potentially improving convergence and performance in various neural network architectures.

**Definition**: The Logish activation function is defined as:

$$
f(x) = x \cdot \ln\left(1 + \sigma(x)\right)
$$

where $\sigma(x)$ is the sigmoid function.

**Derivative:**

$$
f'(x) = \ln\left(1 + \sigma(x)\right) + x \cdot \frac{\sigma(x) \cdot (1 - \sigma(x))}{1 + \sigma(x)}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'logish'}); 
```


### Softmin
Softmin is an activation function that applies the softmin operation to its inputs, effectively transforming them into a probability distribution where smaller values are amplified. It is particularly useful for tasks where the goal is to emphasize smaller input values.

**Definition**: The Softmin activation function is defined as:

$$
f(x_i) = \frac{e^{-x_i}}{\sum_{j} e^{-x_j}}
$$

**Derivative:**

$$
\frac{\partial f(x_i)}{\partial x_j} = \begin{cases}
f(x_i) \cdot (1 - f(x_i)) & \text{if } i = j \\
-f(x_i) \cdot f(x_j) & \text{if } i \neq j
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'softmin', num_classes:10});
```

### Softsign
Softsign is a smooth and differentiable activation function that approximates the sign function with a soft transition. This function provides a continuous approximation of the sign function, helping to mitigate the problem of vanishing gradients and improving the learning dynamics in neural networks.

**Definition**: The Softsign activation function is defined as:

$$
f(x) = \frac{x}{1 + |x|}
$$

**Derivative:**

$$
f'(x) = \frac{1}{(1 + |x|)^2}
$$

**Example:**

```javascript
layer_defs.push({type:'softsign', num_classes:10});
```

### Softshrink
Softshrink is a thresholding activation function that introduces sparsity by shrinking values towards zero. This function is useful for regularization and feature selection, as it helps to reduce the impact of small values and promote sparsity in the activations.

**Definition**: The Softshrink activation function is defined as:

$$
f(x) = \begin{cases} 
x - \lambda, & \text{if } x > \lambda \\
x + \lambda, & \text{if } x < -\lambda \\
0, & \text{otherwise}
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1, & \text{if } |x| > \lambda \\
0, & \text{otherwise}
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'softshrink', num_classes:10, lambda: 0.5});
```

### Hardshrink
Hardshrink is a simple threshold-based activation function that sets values within a specific range to zero. This function is effective in scenarios where you want to introduce sparsity and handle outliers by zeroing out values within a certain range.

**Definition**: The Hardshrink activation function is defined as:

$$
f(x) = \begin{cases} 
x, & \text{if } |x| > \lambda \\
0, & \text{otherwise } 
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1, & \text{if } |x| > \lambda \\
0, & \text{otherwise}
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'hardshrink', num_classes:10, lambda: 0.5});
```

### GeLU
Gaussian Error Linear Unit (GeLU) is an activation function that applies a smooth curve to the input values, offering a balance between linearity and non-linearity. It has shown superior performance in various neural network architectures, particularly in natural language processing tasks.

**Definition**: The Gaussian Error Linear Unit (GeLU) activation function is defined as:

$$
f(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

**Derivative:**

$$
f'(x) = \Phi(x) + x \cdot \phi(x)
$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution, and $\phi(x)$ is the probability density function of the standard normal distribution.

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'gelu'}); 
```


### PReLU
Parametric Rectified Linear Unit (PReLU) is an activation function that introduces learnable parameters for the negative slope. It allows the model to adapt the negative part of the function during training, which can improve the model's capacity and performance.

**Definition**: The Parametric Rectified Linear Unit (PReLU) activation function is defined as:

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'prelu', alpha: 0.01}); 
```


### RReLU
Randomized Leaky Rectified Linear Unit (RReLU) is an activation function that introduces randomness to the negative slope during training, which can act as a form of regularization and help prevent overfitting. The slope is chosen from a uniform distribution within a given range.

**Definition**: The Randomized Leaky Rectified Linear Unit (RReLU) activation function is defined as:

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

where $\alpha$ is a random variable sampled from a uniform distribution $\alpha \sim \text{Uniform}(l, u)$ during training, and fixed during inference.

**Derivative:**

$$
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0 
\end{cases}
$$

**Example:**

```javascript
layer_defs.push({type:'fc', num_neurons:20, activation:'rrelu', lower: 0.01, upper: 0.1}); 
```

## TODO

Some activation functions/loss functions that can be added in the future:

- Smish
- LogSoftmax





