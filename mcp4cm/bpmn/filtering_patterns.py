import re

# Threshold for near-duplicate detection
TFIDF_DUPLICATE_THRESHOLD = 0.8  # Threshold for TF-IDF similarity (0.0-1.0)
MIN_ELEMENT_COUNT = 5
MAX_ELEMENT_COUNT = 200
MAX_EMPTY_NAME_PERCENTAGE = 0.35
DUMMY_WORD_THRESHOLD = 0.6
MIN_MEDIAN_NAME_LENGTH = 3
DUPLICATE_ACTIVITY_NAME_THRESHOLD = 0.4 # If 40% or more activities have duplicate names, most likely bad model
                                        # rule of best practice: do not name multiple activities with the same name

MINIMAL_ELEMENTS_DICT = {'Activity': 1, 'StartEvent': 1, 'EndEvent': 1, 'SequenceFlow': 2}

START_EVENT_PATTERN = re.compile(r"^Start\w*Event$")
END_EVENT_PATTERN = re.compile(r"^End\w*Event$")
EVENT_PATTERN = re.compile(r"^\w*Event$")
DATA_OBJECT_PATTERN = re.compile(r"^DataObject$")
GATEWAY_PATTERN = re.compile(r"^\w*Gateway$")
FLOW_PATTERN = re.compile(r"^\w*Flow$")
SEQUENCE_FLOW_PATTERN = re.compile(r"^SequenceFlow$")
ASSOCIATION_PATTERN = re.compile(r"Association\w*$")
TASK_PATTERN = re.compile(r"^Task\w*$")
SUBPROCESS_PATTERN = re.compile(r"^\w*Subprocess$")
ACTIVITY_PATTERN = re.compile(r"^Task|\w*Subprocess$")
EMPTY_NAME_PATTERN = re.compile(r"^empty name$")



DUMMY_KEYWORDS = {'task', 'pool', 'lane', 'department', 'company', 'activity', 'start', 'end'}

