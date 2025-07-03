"""
MCP4CM - Model Cleansing Package for Conceptual Models

A comprehensive library for cleaning and preprocessing conceptual model datasets,
with a focus on UML models.

The package provides tools for:
- Loading and parsing UML model datasets
- Cleaning models by filtering out empty or invalid files
- Detecting and removing duplicate models
- Filtering models based on naming patterns and quality metrics
- Language detection for model content
- Extracting metadata and statistical information from models
"""

__version__ = "1.0.1"

from mcp4cm.uml.data_extraction import (
    filter_classes_by_generic_pattern as uml_filter_classes_by_generic_pattern,
    filter_dummy_classes as uml_filter_dummy_classes,
    filter_dummy_names as uml_filter_dummy_names,
    filter_empty_or_invalid_files as uml_filter_empty_or_invalid_files,
    filter_models_by_name_count as uml_filter_models_by_name_count,
    filter_models_by_name_length_or_stopwords as uml_filter_models_by_name_length_or_stopwords,
    filter_models_by_sequential_and_dummy_words as uml_filter_models_by_sequential_and_dummy_words,
    filter_models_with_empty_class_names as uml_filter_models_with_empty_class_names,
    filter_models_without_names as uml_filter_models_without_names,
    find_files_with_comments as uml_find_files_with_comments
)

from mcp4cm.dataloading import (
    load_dataset
)

# List of all UML filtering functions for easy access
uml_filters = [
    uml_filter_classes_by_generic_pattern,
    uml_filter_dummy_classes,
    uml_filter_dummy_names,
    uml_filter_empty_or_invalid_files,
    uml_filter_models_by_name_count,
    uml_filter_models_by_name_length_or_stopwords,
    uml_filter_models_by_sequential_and_dummy_words,
    uml_filter_models_with_empty_class_names,
    uml_filter_models_without_names,
    uml_find_files_with_comments
]
 

__all__ = [
    "__version__",
    "uml_filter_classes_by_generic_pattern",
    "uml_filter_dummy_classes",
    "uml_filter_dummy_names",
    "uml_filter_empty_or_invalid_files",
    "uml_filter_models_by_name_count",
    "uml_filter_models_by_name_length_or_stopwords",
    "uml_filter_models_by_sequential_and_dummy_words",
    "uml_filter_models_with_empty_class_names",
    "uml_filter_models_without_names",
    "uml_find_files_with_comments",
    "uml_filters",
    
    "load_dataset"
]
