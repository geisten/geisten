
# geisten - The library for creating efficient binary neural networks

Author: Germar Schlegel

Date: 01.09.2021

This c header only library provides an efficient network architecture
and a set of two hyper-parameters in order to build very
small, low latency models that can be easily matched to the
design requirements for mobile and embedded vision applications.

The 1-bit convolutional neural network (1-bit CNN, also known as binary neu-
ral network) [7,30], of which both weights and activations are binary, has
been recognized as one of the most promising neural network compression
methods for deploying models onto the resource-limited devices. It enjoys 32×
memory compression ratio, and up to 58×practical computational reduction on
CPU, as demonstrated in [30]. Moreover, with its pure logical computation
(i.e., XNOR operations between binary weights and binary activations), 1-bit
CNN is both highly energy-efficient for embedded devices [8,40], and
possesses the potential of being directly deployed on next generation
memristor-based hardware.

The library is based on following papers:
[ReActNet: Towards Precise Binary Neural Network with Generalized Activation
Functions](https://arxiv.org/pdf/2003.03488.pdf)

## TODO
- Create loss function
- Create softmax function
- Optimize weights
-


The special operator __has_builtin (operand) is used to test whether
the symbol named by its operand is recognized as a built-in function by GCC
in the current language and conformance mode. The __has_builtin operator by
itself, without any operand or parentheses, acts as a predefined macro so
that support for it can be tested in portable code.


rprelu_fp16() - RPReLU function
- `x` The function variable
- `beta` The slope constant, if x <= gamma
- `gamma` The level constant
- `zeta` The offset constant

[Background](https://arxiv.org/pdf/2003.03488.pdf)


## rprelu_derived() - The derived RPReLU function
- `x` The function variable
- `beta` The slope constant, if x <= gamma
- `gamma` The level constant

_Returns the derived function value_


## binarize() - Binarizes 8 bit fix pont elements of array `x`
- `x` The fix point array of size BIT_SIZE(unsigned long long)
- `threshold` The conversion threshold

Binarizes all elements of array `x` and writes the result to the bit array
`b`. The conversion is as follows:

    if x[i] > threshold then set bit=1 else set bit=0

Returns the bit array (of size BIT_SIZE(unsigned long long))


## delta() - Calculate the error between the output vector and the target
- `m` The number of vector elements
- `a` The output vector
- `y` The target vector
- `d` The calculated error (delta) vector


## forward() - Linear forward transformation
- `m` The number of input cells (weight matrix rows)
- `wb` The `j`th column of the binary m x n weight matrix
- `alpha` The binarization constant
- `x` The activation vector

### Bitwise matrix multiplication

`xb` and `wb[i]` are both binary values thus  `xb ∗ wb[i]` can be implemented with bitwise
operations. In each bit, the result of basic multiplication `xb × wb[i]` is one of
three values {−1,0,1}.
(Details)[https://arxiv.org/pdf/1909.11366.pdf]

Return the sum of the element wise multiplication.


## backward() - Backward propagation
- `m` The number of input cells (weight matrix rows)
- `wb` The `j`th column of the binary m x n weight matrix
- `y` The output cell at position `j`
- `x` The input vector (of size `m`)

The resulting x vector must also be multiplied by the weight factor to obtain
the correct value.


## update_weights() - Update the weight matrix
- `m` The number of input cells (weight matrix rows)
- `x` The input vector (of size `m`) 
- `delta` The output cell value (delta y) at position `j`
- `w` The binary m x n weight matrix to be updated

The function updates a column of a binary matrix.
The matrix is binarized around the 0 value. 
If the value is greater than 0, then the updated binary 
matrix is set to 1; otherwise set to 0 
The learning rate can be controlled via the delta value 
(e.g.: delta_learn = delta * rate)


