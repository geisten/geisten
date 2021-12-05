<h1 align="center">
   <img src="./img/neuron.png" alt="geisten neurons">
</h1>
<h4 align="center"><b>The minimal C header only binary network library you are looking for</b></h4>

---

![GitHub last commit](https://img.shields.io/github/last-commit/geisten/geisten?style=plastic)
![Lines of code](https://img.shields.io/tokei/lines/github/geisten/geisten?style=plastic)
![GitHub issues](https://img.shields.io/github/issues/geisten/geisten?style=plastic)
![GitHub](https://img.shields.io/github/license/geisten/geisten?style=plastic)

---
geisten is a low power tiny binary network library written in **C**. The library is designed for _"scalability, speed
and ease-of-use"_ as well as optimized for Unix/POSIX (*BSD, Linux, MacOSX) like environments.

## Getting Started

Since the library is a header-only C library, the file [geisten.h](geisten.h) can simply be copied to the desired
location and included into your source code build path. The header file can also be copied to /usr/include/ using:

```shell
make install
```

With the `DESTDIR` and `PREFIX` environment variables you can adjust the destination. The destination path is set in the
form: `$(DESTDIR)$(PREFIX)/include/`. The path must already exist.

```shell
DESTDIR=/home/germar make install
```

## Differences to existing frameworks

geisten is a minimalistic neural network written in C. In contrast to Keras, Tensorflow, etc. geisten is much smaller
and simpler.

- strongly reduced functionality compared to the aforementioned alternatives
- geisten has no Lua integration, no shell-based configuration and comes without any additional tools.
- geisten is only a single binary, and the source code of the reference implementation is intended to never exceed 500
  SLOC.
- geisten is customized through editing its source code, which makes it extremely fast and secure - it does not process
  any input data which isn't known at compile time. You don't have to activate Lua/Python or some weird configuration
  file format, beside C, to customize it for your needs: you only have to activate C (at least in order to edit the
  header file).
- Because geisten is customized through editing its source code, it's pointless to make binary packages of it.

## Documentation

See [geisten.md](geisten.md) for the function description of the library. It uses a little script (xtract.awk) from
the [d.awk project](https://github.com/wernsey/d.awk) to extract the function documentation from the geisten header
file.

## Support

The first step is to look at the source code for obvious names, which could be related to the problem that arose.

## Development

geisten is actively developed. You can browse its [source code](https://github.com/geisten/geisten.git) repository or
get a copy using git with the following command:

```shell
git clone https://github.com/geisten/geisten.git
```

## Feedback

If you have any feedback, please reach out to us at g.schlegel@geisten.com
