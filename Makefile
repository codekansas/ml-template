# Makefile

define HELP_MESSAGE
                    ML Template
                    -----------

# Installing

1. Create a new Conda environment: `conda create --name ml-template python=3.8`
2. Activate the environment: `conda activate ml-template`
3. Install dependencies: `make install-deps`
4. Install the package: `make install`

# Running Tests

1. Install dependencies: `make install-format-deps`
2. Run autoformatting: `make format`
3. Run static checks: `make static-checks`
4. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#      Configuration      #
# ------------------------ #

# Configuration options.
TORCH_VERSION_BASE := 1.12.1
TORCHVISION_VERSION_BASE := 0.13.1
CUDA_VERSION := cu116

# Detects backend.
ifdef CUDA_HOME
BACKEND := cuda
else ifeq ($(shell uname -p), arm)
BACKEND := metal
else
BACKEND := cpu
endif

# Conda channels.
CONDA_CHANNELS =
CONDA_CHANNELS += -c conda-forge

# Conda dependencies.
CONDA_DEPS =
CONDA_DEPS += cmake=3.23.1
CONDA_DEPS += zstd=1.5.2

# Conda formatting channels.
CONDA_FORMAT_CHANNELS =
CONDA_FORMAT_CHANNELS += -c conda-forge

# Conda formatting dependencies.
CONDA_FORMAT_DEPS =
CONDA_FORMAT_DEPS += clang-format=14.0.4
CONDA_FORMAT_DEPS += cmake-format=0.6.11

# Pip formatting dependencies.
PIP_FORMAT_DEPS =
PIP_FORMAT_DEPS += black
PIP_FORMAT_DEPS += darglint
PIP_FORMAT_DEPS += flake8
PIP_FORMAT_DEPS += mypy-extensions
PIP_FORMAT_DEPS += mypy
PIP_FORMAT_DEPS += pylint
PIP_FORMAT_DEPS += pytest
PIP_FORMAT_DEPS += types-setuptools
PIP_FORMAT_DEPS += typing_extensions

# Gets the Torch version to install.
ifeq ($(BACKEND),cuda)
TORCH_VERSION := $(TORCH_VERSION_BASE)+$(CUDA_VERSION)
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)+$(CUDA_VERSION)
else ifeq ($(BACKEND),metal)
TORCH_VERSION := $(TORCH_VERSION_BASE)
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)
else ifeq ($(BACKEND),cpu)
TORCH_VERSION := $(TORCH_VERSION_BASE)+cpu
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)+cpu
else
$(error "Unsupported backend: $(BACKEND)")
endif

PIP_ARGS =
PIP_ARGS += --find-links https://download.pytorch.org/whl/torch_stable.html

# ------------------------ #
#        Initialize        #
# ------------------------ #

PLATFORM := python -c 'import platform; print(platform.system())'

# Ensures that the user is in an Anaconda environment,
# and that Mamba is installed.
initialize:
ifeq (, $(shell which mamba))
ifeq (, $(shell which conda))
	$(error Conda is not installed)
else ifeq (, $(CONDA_DEFAULT_ENV))
	$(error Conda not installed or not initialized)
else ifeq (base, $(CONDA_DEFAULT_ENV))
	$(error Don't install this package into the base environment. Run 'conda create --name ml python=3.8' then 'conda activate ml`)
else
	conda install -c conda-forge mamba
endif
endif
.PHONY: initialize

# ------------------------ #
#          Build           #
# ------------------------ #

install-deps: initialize
	@mamba install $(CONDA_CHANNELS) $(CONDA_DEPS)
.PHONY: install-dependencies

install: initialize
	@TORCH_VERSION=$(TORCH_VERSION) TORCHVISION_VERSION=$(TORCHVISION_VERSION) pip install $(PIP_ARGS) -e .
.PHONY: install

clean:
	rm -rf build/ **/*.egg-info **/*.pyc **/*.so ml/**/*.pyi ml/**/*.so
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

py-files := $$(git ls-files '*.py')
cpp-files := $$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh')
cmake-files := $$(git ls-files '*/CMakeLists.txt')

install-format-deps: initialize
	@mamba install $(CONDA_FORMAT_CHANNELS) $(CONDA_FORMAT_DEPS)
	@pip install $(PIP_FORMAT_DEPS)

format: initialize
	cmake-format -i $(cmake-files)
	clang-format -i $(cpp-files)
	black $(py-files)
	isort $(py-files)
.PHONY: format

static-checks: initialize
	black --diff --check $(py-files)
	isort --check-only $(py-files)
	mypy --install-types --non-interactive $(py-files)
	flake8 --count --show-source --statistics $(py-files)
	pylint $(py-files)
.PHONY: lint

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	pytest .
.PHONY: test
