"""
Filtering patterns and thresholds for UML model cleaning.

This module defines regular expressions, keywords, and threshold values used
for identifying and filtering low-quality or auto-generated UML models.
"""

import re

# Regular expressions for name pattern matching
dummy_name_pattern = re.compile(r'^att(\s+[A-Za-z]|\s+\d+|[a-z0-9])?$', re.IGNORECASE)  # Pattern for dummy attribute names
dummy_class_pattern = re.compile(r'^class\s?[a-z0-9]$', re.IGNORECASE)  # Strictly match 'class 1', 'class a', etc.

# Threshold values used for filtering models
DUMMY_WORD_THRESHOLD = 0.82  # % of names that are generic dummy words
GENERIC_PATTERN_THRESHOLD_COUNT = 2  # % of names that match a generic pattern
DUMMY_NAMES_THRESHOLD = 0.3  # % of class names that are dummy names
MIN_SHORT_NAME_LENGTH = 2  # Minimum length for short names
SHORT_NAMES_UPPER_THRESHOLD = 0.5  # % of names that are short names

