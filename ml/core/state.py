from dataclasses import dataclass
from typing import Literal, Tuple, TypeVar, cast, get_args

from omegaconf import MISSING
from torch import nn

from ml.core.config import conf_field

Module = TypeVar("Module", bound=nn.Module)

Phase = Literal["train", "valid", "test"]


def set_phase(model: Module, phase: Phase) -> Tuple[Module, Phase]:
    if phase == "train":
        if not model.training:
            model = model.train()
        return model, phase
    else:
        if model.training:
            model = model.eval()
        return model, phase


def cast_phase(raw_phase: str) -> Phase:
    args = get_args(Phase)
    assert raw_phase in args, f"Invalid phase: '{raw_phase}' Valid options are {args}"
    return cast(Phase, raw_phase)


@dataclass
class State:
    """Defines the state variables to track training."""

    num_epochs: int = conf_field(MISSING, help="Number of epochs so far")
    num_steps: int = conf_field(MISSING, help="Number of steps so far")
    num_samples: int = conf_field(MISSING, help="Number of sample so far")
    num_valid_steps: int = conf_field(MISSING, help="Number of validation steps so far")
    num_test_steps: int = conf_field(MISSING, help="Number of test steps so far")
    raw_phase: str = conf_field(MISSING, help="Current training phase")

    @property
    def phase(self) -> Phase:
        return cast_phase(self.raw_phase)

    @phase.setter
    def phase(self, new_phase: Phase) -> None:
        self.raw_phase = new_phase

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            num_epochs=0,
            num_steps=0,
            num_samples=0,
            num_valid_steps=0,
            num_test_steps=0,
            raw_phase="train",
        )

    @property
    def training(self) -> bool:
        return self.phase == "train"
