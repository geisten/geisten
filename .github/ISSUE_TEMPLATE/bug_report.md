---
name: Bug report
about: Something built or ran incorrectly
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug.

**Expected**
What you expected instead.

**Repro**
Exact commands. Include the model if relevant.
```sh
make TARGET=...
...
```

**Environment**
- geist version / commit:
- Target + mode (e.g. `mac-omp` / `release`):
- Host (CPU, OS): `uname -a`
- Compiler: `cc --version`
- Model (GGUF name + quant):

**Logs**
Relevant stderr/stdout. For crashes, a `MODE=asan` build trace helps a lot.
