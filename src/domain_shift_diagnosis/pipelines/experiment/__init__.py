"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""

from .pipeline import create_pipeline
from .baseline_pipeline import create_pipeline as create_baseline_pipeline

__all__ = ["create_pipeline", "create_baseline_pipeline"]
