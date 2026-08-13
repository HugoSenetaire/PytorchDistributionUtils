# PytorchDistributionUtils

PyTorch utilities for working with probability distributions, gradient estimators, and distribution wrappers.

## Overview

A lightweight utility library providing:

- **Custom distributions** beyond what `torch.distributions` offers
- **Gradient estimators** for expectations over discrete and continuous distributions
- **Wrappers** for composing and transforming distributions

## Structure

```
distribution/         # Custom probability distribution implementations
gradientestimator/    # Gradient estimators (REINFORCE, reparameterisation, etc.)
wrappers/             # Distribution wrappers and transformations
```

## Installation

```bash
git clone https://github.com/HugoSenetaire/PytorchDistributionUtils.git
```

Then add to your project:

```python
import sys
sys.path.append("path/to/PytorchDistributionUtils")
from distribution import ...
```

## Requirements

- PyTorch
