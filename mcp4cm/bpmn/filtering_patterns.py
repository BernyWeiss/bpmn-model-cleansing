








# Threshold for near-duplicate detection
TFIDF_DUPLICATE_THRESHOLD = 0.8  # Threshold for TF-IDF similarity (0.0-1.0)
MIN_ELEMENT_COUNT = 5
MAX_ELEMENT_COUNT = 200
MAX_EMPTY_NAME_PERCENTAGE = 0.35
DUMMY_WORD_THRESHOLD = 0.6
MIN_MEDIAN_NAME_LENGTH = 4


# TODO: convert placeholders into actual pattern to use in filters and name extraction
ACTIVITY_TYPES = ('Task', '...Subprocess')
START_EVENT_PATTERN = 'Start...Event'
END_EVENT_PATTERN = 'End...Event'
EVENT_PATTERN = '...Event'
DATA_OBJECT_PATTERN = 'DataObject'
GATEWAY_PATTERN = '...Gateway'
FLOW_PATTERN = '...Flow'
ASSOCIATION_PATTERN = 'Association...'


DUMMY_KEYWORDS = ['empty name', 'task', 'pool', 'lane', 'department', 'company']