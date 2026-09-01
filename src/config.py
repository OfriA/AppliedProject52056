"""Central configuration for the CBCL text-severity prediction project."""

RANDOM_STATE = 42

TEXT_COLUMNS = ["Event", "EER_text"]
TARGET_COLUMN = "CBCL_score"

LOW_Q = 0.50
HIGH_Q = 0.85

TEST_SIZE = 0.20

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
