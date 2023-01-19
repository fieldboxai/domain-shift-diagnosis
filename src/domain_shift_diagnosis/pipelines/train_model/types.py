from typing import TypedDict


class TrainingParams(TypedDict):
    loss: str
    batch_size_ratio: float
    learning_rate: float
    tol: float
    patience: float
    random_state: int
    epochs: int
    monitor: str
