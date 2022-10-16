# ML Project Template

This is a general-purpose template for machine learning projects in PyTorch. It includes a simple CIFAR example which can be deleted.

## Getting Started

### Installation

First, create a Conda environment:

```bash
conda create --name ml python=3.10
```

Next, install the project dependencies:

```bash
make install-dependencies
```

Finally, install the project (in editable mode):

```bash
make install
```

### Your First Command

Run the CIFAR training example:

```bash
ml train configs/cifar_demo.yaml
```

Launch a Slurm job (requires setting the `SLURM_PARTITION` environment variable):

```bash
ml mp_train configs/cifar_demo.yaml trainer.name=slurm
```

### Architecture

A new project is broken down into five parts:

1. *Task*: Defines the dataset and calls the model on a sample. Similar to a [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).
2. *Model*: Just a PyTorch `nn.Module`
3. *Trainer*: Defines the main training loop, and optionally how to distribute training when using multiple GPUs
4. *Optimizer*: Just a PyTorch `optim.Optimizer`
5. *LR Scheduler*: Just a PyTorch `optim.LRScheduler`

Most projects should just have to implement the Task and Model, and use a default trainer, optimizer and learning rate scheduler. Running the training command above will log the location of each component.

## Features

This repository implements some features which I find useful when starting ML projects.

### C++ Extensions

This template makes it easy to add custom C++ extensions to your PyTorch project. The demo includes a custom TorchScript-compatible nucleus sampling function, although more complex extensions are possible.

- [Custom TorchScript Op Tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
- [PyTorch CMake Extension Reference](https://github.com/pytorch/extension-script)

### Github Actions

This template automatically runs `black`, `isort`, `pylint` and `mypy` against your repository as a Github action. You can enable push-blocking until these tests pass.

### Lots of Timers

The training loop is pretty well optimized, but sometimes you can do stupid things when implementing a task that impact your performance. This adds a lot of timers which make it easy to spot likely training slowdowns, or you can run the full profiler if you want a more detailed breakdown.
