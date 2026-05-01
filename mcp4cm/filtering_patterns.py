"""
Filtering patterns and thresholds for UML model cleaning.

This module defines regular expressions, keywords, and threshold values used
for identifying and filtering low-quality or auto-generated UML models.
"""

import re

# Regular expressions for name pattern matching
dummy_name_pattern = re.compile(r'^att(\s+[A-Za-z]|\s+\d+|[a-z0-9])?$', re.IGNORECASE)  # Pattern for dummy attribute names
numbered_pattern = re.compile(r'(.+?)[\s_]?(\d+)$', re.IGNORECASE)  # Pattern for sequentially numbered elements


# Threshold for near-duplicate detection
TFIDF_DUPLICATE_THRESHOLD = 0.8  # Threshold for TF-IDF similarity (0.0-1.0)
