"""
GeoguessrBot Inference Engine - Python bindings for Rust-based vector search.

This module provides fast geolocation inference using Milvus vector database.
"""

from .geoguessr_inference import (
    infer_location,
    infer_batch,
    Prediction,
    InferenceResult,
)

__all__ = [
    "infer_location",
    "infer_batch", 
    "Prediction",
    "InferenceResult",
]

__version__ = "0.1.0"
