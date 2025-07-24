"""
Filtering patterns and thresholds for UML model cleaning.

This module defines regular expressions, keywords, and threshold values used
for identifying and filtering low-quality or auto-generated UML models.
"""

import re

# Regular expressions for name pattern matching
empty_name_pattern = r"empty name"  # Pattern for empty names
empty_class_name_pattern = r"class: empty name"  # Pattern for empty class names
comment_pattern = r"comment:"  # Pattern for comments in models
dummy_name_pattern = re.compile(
    r"^att(\s+[A-Za-z]|\s+\d+|[a-z0-9])?$", re.IGNORECASE
)  # Pattern for dummy attribute names
dummy_class_pattern = re.compile(
    r"^class\s?[a-z0-9]$", re.IGNORECASE
)  # Strictly match 'class 1', 'class a', etc.
general_class_pattern = re.compile(r"^[a-z]+", re.IGNORECASE)  # Match any class name
myclass_pattern = re.compile(
    r'^class:\s*my class\s?(\d+)?$', re.IGNORECASE
)  # Pattern for "my class" template names

numbered_pattern = re.compile(
    r"(.+?)[\s_]?(\d+)$", re.IGNORECASE
)  # Pattern for sequentially numbered elements

two_char_pattern = re.compile(r'^[a-zA-Z]\d$', re.IGNORECASE)  # Matches "a1", "B2", etc.
letter_space_letter_pattern = re.compile(r'^[a-zA-Z]\s[a-zA-Z]$', re.IGNORECASE)  # Matches "a b", "x y", etc.


# Set of generic or placeholder terms often found in auto-generated models
DUMMY_KEYWORDS = {
    "my class",  # Generic class name
    "class",  # Placeholder for class
    "use case",  # Generic use case
    "actor",  # Generic actor
    "attribute",  # Generic attribute
    "association",  # Generic association
    "control flow",  # Generic control flow
    "activity",  # Generic activity
    "decision node",  # Generic decision node
    "opaque action",  # Generic action
    "lifeline",  # Generic lifeline
    "flow final node",  # Generic final node
    "activity final node",  # Generic activity final
    "join node",  # Generic join node
    "fork node",  # Generic fork node
    "initial node",  # Generic initial node
    "merge node",  # Generic merge node
    "action",  # Generic action
    "component",  # Generic component
    "ext point",  # Generic extension point
    "empty name",  # Empty name placeholder
    "package",  # Generic package
}

# Threshold values used for filtering models
SEQUENTIAL_THRESHOLD = 0.75  # % of names that follow a sequential pattern
DUMMY_WORD_THRESHOLD = 0.82  # % of names that are generic dummy words
SHORT_DUMMY_WORD_THRESHOLD = 0.3 # % of names that are dummy words shorter than 2 characters
MIN_MEDIAN_SHORT_NAME_LENGTH = 4  # Minimum median length for short names
MIN_NAMES_COUNT = 5  # Minimum number of names per model
VOCABULARY_UNIQUENESS_THRESHOLD = 3  # Minimum unique words per model
GENERIC_PATTERN_THRESHOLD_COUNT = 2  # % of names that match a generic pattern
DUMMY_CLASSES_THRESHOLD = 0.5  # % of class names that are dummy classes
DUMMY_NAMES_THRESHOLD = 0.3  # % of class names that are dummy names
SHORT_NAMES_UPPER_THRESHOLD = 0.30  # % of names that are shorter than 3 characters
SHORT_NAMES_LOWER_THRESHOLD = 0.25  # % of names that are longer than 3 characters
STOPWORDS_THRESHOLD = 0.4  # % of names that are stopwords
MIN_SHORT_NAME_LENGTH = 2  # Minimum length for short names

# Threshold for near-duplicate detection
TFIDF_DUPLICATE_THRESHOLD = 0.8  # Threshold for TF-IDF similarity (0.0-1.0)

# List of frequently occurring names that might indicate auto-generated content
FREQUENCT_NAMES = ["control flow", "control-flow"]
