# ML Project Template

This is a general-purpose template for machine learning projects in PyTorch. It includes a simple CIFAR example which can be deleted.

## Getting Started

### Installation

First, create a Conda environment:

```bash
conda create --name ml python=3.8
```

Next, install the project dependencies:

```bash
make install-deps
```

Finally, install the project:

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

New tasks, models, trainers, optimizers and learning rate schedulers are added using the same API, although each should implement different things. For example, to create a new model, make a new file under `ml/models` and add the following code:

```python
from dataclasses import dataclass

from ml.core.config import conf_field
from ml.core.registry import register_model
from ml.models.base import BaseModel, BaseModelConfig


@dataclass
class NewModelConfig(BaseModelConfig):
  some_param: int = conf_field(10)


@register_model("new_model", NewModelConfig)
class NewModel(BaseModel[NewModelConfig]):
  def forward(self, x):
    return x + self.config.some_param
```

The framework will automatically search in all of the files in `ml/models` to populate the model registry. In your config file, you can then reference the registered model using whatever key you chose:

```yaml
model:
  name: new_model
```

Similar APIs exist for tasks, trainers, optimizers and learning rate schedulers. Try running the demo config to get a sense for how each of these fit together.

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
