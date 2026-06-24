class Split:
    TEST = "test"
    TRAIN = "train"
    DEV = "dev"
    PRED = "predict"


class Benchmark:
    BRIGHT = "BRIGHT"
    BEIR = "beir"
    FRESHSTACK = "freshstack"
    TOOLRET = "toolret"
    SKILLRET = "skillret"
    SRA_BENCH = "sra_bench"


class Dataset:
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
    LANGCHAIN = "langchain"
    YOLO = "yolo"
    ANGULAR = "angular"
    LARAVEL = "laravel"
    GODOT = "godot"
    AMBIGQA = "ambignq"
    WIKIMULTIHOPQA = "wikimultihopqa"
    FINQA = "finqa"
    TATQA = "tatqa"
    MUSIQUEQA = "musiqueqa"
    OTTQA = "ottqa"
    STRATEGYQA = "strategyqa"
    TOOLRET_APIBANK = "apibank"
    TOOLRET_GORILLA_TENSOR = "gorilla-tensor"
    TOOLRET_APPBENCH = "appbench"
    TOOLRET_GORILLA_HF = "gorilla-huggingface"
    TOOLRET_METATOOL = "metatool"
    TOOLRET_RESTGPT_TMDB = "restgpt-tmdb"
    TOOLRET_GPT4TOOLS = "gpt4tools"
    TOOLRET_GTA = "gta"
    TOOLRET_MNMS = "mnms"
    TOOLRET_CRAFT_MATH = "craft-math-algebra"
    TOOLRET_CRAFT_TABMWP = "craft-tabmwp"
    TOOLRET_CRAFT_VQA = "craft-vqa"
    TOOLRET_GORILLA_PYTORCH = "gorilla-pytorch"
    TOOLRET_RESTGPT_SPOTIFY = "restgpt-spotify"
    TOOLRET_TOOLALPACA = "toolalpaca"
    TOOLRET_ULTRATOOL = "ultratool"
    SRA_TOOLQA = "toolqa"
    SRA_THEOREMQA = "theoremqa"
    SRA_BIGCODEBENCH = "bigcodebench"
    SRA_CHAMP = "champ"
    SRA_LOGICBENCH = "logicbench"
    SRA_MEDCALCBENCH = "medcalcbench"


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
FRESHSTACK_TOPICS = [
    Dataset.LANGCHAIN,
    Dataset.YOLO,
    Dataset.ANGULAR,
    Dataset.LARAVEL,
    Dataset.GODOT,
]
TOOLRET_SUBSETS = [
    Dataset.TOOLRET_APIBANK,
    Dataset.TOOLRET_GORILLA_TENSOR,
    Dataset.TOOLRET_APPBENCH,
    Dataset.TOOLRET_GORILLA_HF,
    Dataset.TOOLRET_METATOOL,
    Dataset.TOOLRET_RESTGPT_TMDB,
    Dataset.TOOLRET_GPT4TOOLS,
    Dataset.TOOLRET_GTA,
    Dataset.TOOLRET_MNMS,
    Dataset.TOOLRET_CRAFT_MATH,
    Dataset.TOOLRET_CRAFT_TABMWP,
    Dataset.TOOLRET_CRAFT_VQA,
    Dataset.TOOLRET_GORILLA_PYTORCH,
    Dataset.TOOLRET_RESTGPT_SPOTIFY,
    Dataset.TOOLRET_TOOLALPACA,
    Dataset.TOOLRET_ULTRATOOL,
]
SRA_BENCH_SUBSETS = [
    Dataset.SRA_TOOLQA,
    Dataset.SRA_THEOREMQA,
    Dataset.SRA_BIGCODEBENCH,
    Dataset.SRA_CHAMP,
    Dataset.SRA_LOGICBENCH,
    Dataset.SRA_MEDCALCBENCH,
]


class Separators:
    TABLE_ROW_SEP = "\n"
    TABLE_COL_SEP = "|"


class DataTypes:
    TABLE = "table"
    TEXT = "text"


class HFDatasets:
    BRIGHT_EXAMPLES = "xlangai/BRIGHT"
    BRIGHT_SUBSET_EXAMPLES = "examples"
    BRIGHT_SUBSET_DOCUMENTS = "documents"
    BRIGHT_SUBSET_LONG_DOCUMENTS = "long_documents"
    FRESHSTACK_QUERIES = "freshstack/queries-oct-2024"
    FRESHSTACK_CORPUS = "freshstack/corpus-oct-2024"
    FRESHSTACK_CORPUS_SPLIT = "train"
    FRESHSTACK_QUERIES_SPLIT = "test"
    FRESHSTACK_IRDS_TEMPLATE = "freshstack/{topic}/{split}"
    BRIGHT_IRDS_TEMPLATE = "bright/{task}/{split}"
    TOOLRET_QUERIES = "mangopy/ToolRet-Queries"
    TOOLRET_TOOLS = "mangopy/ToolRet-Tools"
    SKILLRET_REPO = "ThakiCloud/SKILLRET"
    SRA_BENCH_REPO = "WeihangSu/SRA-Bench"
