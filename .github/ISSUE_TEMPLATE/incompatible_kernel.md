---
name: Unsupported or INCOMPATIBLE kernel
about: The tool declined to judge your kernel, and you think it should have.
title: "INCOMPATIBLE: "
labels: compatibility
assignees: ''
---

<!--
INCOMPATIBLE, INCORRECT and ERROR are three different things here, on purpose.

  INCOMPATIBLE — the tool could not judge this kernel. Not a verdict about
                 your code. That is this template.
  INCORRECT    — it judged your kernel and your kernel is wrong.
                 Use "Disputed measurement" if you think that call is wrong.
  ERROR        — the tool itself fell over. Use "Bug report".

If you got INCORRECT or a traceback, one of the other templates fits better.
-->

### What you ran

```bash

```

### What it said

<!-- Paste the INCOMPATIBLE line and the reason it gave, if it gave one. -->

```

```

### What the kernel does

<!--
Describe the signature and the operation. Kernel source is welcome but never
required — this tool exists so that source does not have to leave your machine.

Useful things to say: how many inputs and outputs, their dtypes, whether shapes
are constrained (powers of two only, minimum size, alignment), whether it
mutates an input in place, whether it needs a non-standard launch grid, and
whether any argument is a compile-time constant.
-->

### Why you think it should be supported

<!--
Is this a common shape of kernel? Is it the pattern most of a real codebase
uses? Concrete beats general: "every attention kernel in our repo does this"
tells us more than "this seems like it should work".
-->

### Environment

- crucible / shapesandstrides version:
- Python:
- torch:
- triton:
- OS:
- GPU and driver version:
