import numpy as np
from typing import Any, Protocol

from .._types import SiameseTrainingPair


class SiameseTrackerTrain_DataTransform(Protocol):
    def __call__(self, training_pair: SiameseTrainingPair, rng_engine: np.random.Generator) -> Any:
        ...
