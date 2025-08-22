from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
# TODO: configure this path
# RAW_DATA_PATH = "Your dataset path here"
RAW_DATA_PATH = "/home/seewon/MELP/data/physionet.org/files/ptb-xl/1.0.3"
SPLIT_DIR = ROOT_PATH / "src/melp/data_split"
PROMPT_PATH = ROOT_PATH / "src/melp/prompt/CKEPE_prompt.json"
DATASET_LABELS_PATH = ROOT_PATH / "src/melp/prompt/dataset_class_names.json"
RESULTS_PATH = ROOT_PATH / "logs/melp/results"
ECGFM_PATH = ""