from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
# TODO: configure this path
# RAW_DATA_PATH = "Your dataset path here"
RAW_DATA_PATH = "/home/mkeoliya/common-data/arpa-h/ca"
PROCESSED_DATA_PATH = "/home/mkeoliya/MELP/data/"
OTHER_DATA_PATH = "/home/mkeoliya/common-data/arpa-h/ca"
SPLIT_DIR = ROOT_PATH / "src/melp/data_split"
PROMPT_PATH = ROOT_PATH / "src/melp/prompt/CKEPE_prompt.json"
DATASET_LABELS_PATH = ROOT_PATH / "src/melp/prompt/dataset_class_names.json"
RESULTS_PATH = ROOT_PATH / "logs/melp/results"
ECGFM_PATH = ""