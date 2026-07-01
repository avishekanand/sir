"""
Constants for SIR data loading pipeline.
"""


class Split:
    TEST = "test"
    TRAIN = "train"
    DEV = "dev"
    PRED = "predict"


class Benchmark:
    BRIGHT = "BRIGHT"
    BEIR = "beir"
    FRESHSTACK = "freshstack"


class Dataset:
    # BRIGHT tasks
    BIOLOGY = "biology"
    EARTH_SCIENCE = "earth_science"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    ROBOTICS = "robotics"
    STACKOVERFLOW = "stackoverflow"
    SUSTAINABLE_LIVING = "sustainable_living"
    LEETCODE = "leetcode"
    PONY = "pony"
    AOPS = "aops"
    THEOREMQA_QUESTIONS = "theoremqa_questions"
    THEOREMQA_THEOREMS = "theoremqa_theorems"

    # FreshStack topics
    LANGCHAIN = "langchain"
    YOLO = "yolo"
    ANGULAR = "angular"
    LARAVEL = "laravel"
    GODOT = "godot"

    # Legacy DEXTER datasets
    AMBIGQA = "ambignq"
    WIKIMULTIHOPQA = "wikimultihopqa"
    FINQA = "finqa"
    TATQA = "tatqa"
    MUSIQUEQA = "musiqueqa"
    OTTQA = "ottqa"
    STRATEGYQA = "strategyqa"


# All BRIGHT task names as a list for easy iteration
BRIGHT_TASKS = [
    Dataset.BIOLOGY,
    Dataset.EARTH_SCIENCE,
    Dataset.ECONOMICS,
    Dataset.PSYCHOLOGY,
    Dataset.ROBOTICS,
    Dataset.STACKOVERFLOW,
    Dataset.SUSTAINABLE_LIVING,
    Dataset.LEETCODE,
    Dataset.PONY,
    Dataset.AOPS,
    Dataset.THEOREMQA_QUESTIONS,
    Dataset.THEOREMQA_THEOREMS,
]

# FreshStack topics
FRESHSTACK_TOPICS = [
    Dataset.LANGCHAIN,
    Dataset.YOLO,
    Dataset.ANGULAR,
    Dataset.LARAVEL,
    Dataset.GODOT,
]


class Separators:
    TABLE_ROW_SEP = "\n"
    TABLE_COL_SEP = "|"


class DataTypes:
    TABLE = "table"
    TEXT = "text"


# HuggingFace dataset identifiers
class HFDatasets:
    BRIGHT_EXAMPLES = "xlangai/BRIGHT"
    BRIGHT_SUBSET_EXAMPLES = "examples"
    BRIGHT_SUBSET_DOCUMENTS = "documents"
    BRIGHT_SUBSET_LONG_DOCUMENTS = "long_documents"

    FRESHSTACK_QUERIES = "freshstack/queries-oct-2024"
    FRESHSTACK_CORPUS  = "freshstack/corpus-oct-2024"
    # The corpus lives in the "train" split on HuggingFace;
    # the queries test/train split maps to the logical evaluation split.
    FRESHSTACK_CORPUS_SPLIT  = "train"
    FRESHSTACK_QUERIES_SPLIT = "test"

    # ir_datasets path templates  (formatted with topic/split)
    FRESHSTACK_IRDS_TEMPLATE = "freshstack/{topic}/{split}"
    BRIGHT_IRDS_TEMPLATE     = "bright/{task}/{split}"
