<h1 align="center">
   <img src="./img/neuron.png" alt="geisten neurons">
</h1>
<h4 align="center">The minimal c header only 1 Bit deep learning library you are looking for</h4>

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20OS%20%7C%20BSD-blue.svg)]()
[![Bugs](https://img.shields.io/github/issues/geisten/geisten.svg)](https://github.com/geisten/geisten/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

---
geisten is a low power tiny binary network library written in **C**. The library is designed for _"scalability, speed and ease-of-use"_ as well as optimized for Unix/POSIX (*BSD, Linux, MacOSX) like environments.

## Getting Started

Since the library is a header-only C library, the file [geisten.h](geisten.h) can simply be copied to the desired location and included into your source code build path. The header file can also be copied to /usr/include/ using:

```shell
make install
```

## Differences to existing frameworks

geisten is a minimalistic neural network written in C.
In contrast to Keras, Tensorflow, etc. geisten is much smaller and simpler.

- strongly reduced functionality compared to the aforementioned alternatives
- geisten has no Lua integration, no shell-based configuration and comes without any additional tools.
- geisten is only a single binary, and the source code of the reference implementation is intended to never exceed 1000 SLOC.
- geisten is customized through editing its source code, which makes it extremely fast and secure - it does not process any input data which isn't known at compile time. You don't have to activate Lua/Python or some weird configuration file format, beside C, to customize it for your needs: you only have to activate C (at least in order to edit the header file).
- Because geisten is customized through editing its source code, it's pointless to make binary packages of it.

## Support

The first step is to look at the source code for obvious names, which could be related to the problem that arose.

## Development

geisten is actively developed. You can browse its [source code](https://github.com/geisten/geisten.git) repository or get a copy using git with the following command:

```shell
git clone https://github.com/geisten/geisten.git
```

## Feedback

If you have any feedback, please reach out to us at g.schlegel@geisten.com
