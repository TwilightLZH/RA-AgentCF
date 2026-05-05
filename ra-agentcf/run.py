import argparse
import importlib
import os
import sys
import warnings
from logging import getLogger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
AGENTCF_ROOT = os.path.join(REPO_ROOT, "agentcf")

for path in [CURRENT_DIR, AGENTCF_ROOT]:
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, AGENTCF_ROOT)
sys.path.insert(0, CURRENT_DIR)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import get_model as recbole_get_model
from recbole.utils import get_trainer, init_logger, init_seed, set_color

from dataset import BPRDataset, ITEMBPRDataset
from trainer import LanguageLossTrainer, RAAgentCFTrainer


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).lower() in {"1", "true", "yes", "y"}


def get_model(model_name):
    module_name = f"model.{model_name.lower()}"
    try:
        model_module = importlib.import_module(module_name)
        return getattr(model_module, model_name)
    except (ImportError, AttributeError):
        return recbole_get_model(model_name)


def run_baseline(model_name, dataset_name, **kwargs):
    ra_dataset_props = os.path.join(CURRENT_DIR, "props", f"{dataset_name}.yaml")
    agentcf_dataset_props = os.path.join(AGENTCF_ROOT, "props", f"{dataset_name}.yaml")
    dataset_props = ra_dataset_props if os.path.exists(ra_dataset_props) else agentcf_dataset_props
    props = [
        os.path.join(AGENTCF_ROOT, "props", "overall.yaml"),
        os.path.join(AGENTCF_ROOT, "props", "AgentCF.yaml"),
        os.path.join(CURRENT_DIR, "props", "RAAgentCF.yaml"),
        dataset_props,
    ]
    config_overrides = dict(kwargs)
    if dataset_props == ra_dataset_props:
        dataset_dir = os.path.join(CURRENT_DIR, "dataset", dataset_name)
        config_overrides.setdefault("data_path", os.path.join(CURRENT_DIR, "dataset"))
        config_overrides.setdefault("record_path", os.path.join(CURRENT_DIR, "dataset"))
        config_overrides.setdefault("ra_revenue_source", "file")
        config_overrides.setdefault("ra_revenue_file", os.path.join(dataset_dir, f"{dataset_name}.revenue.csv"))
        config_overrides.setdefault("ra_item_behavior_file", os.path.join(dataset_dir, f"{dataset_name}.item_behavior.csv"))
        config_overrides.setdefault("ra_user_profile_file", os.path.join(dataset_dir, f"{dataset_name}.user_profile.csv"))
    else:
        config_overrides.setdefault("data_path", os.path.join(AGENTCF_ROOT, "dataset"))
        config_overrides.setdefault("record_path", os.path.join(AGENTCF_ROOT, "dataset"))
    model_class = get_model(model_name)
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=config_overrides,
    )
    debug = parse_bool(getattr(config, "final_config_dict", {}).get("debug", False))
    if debug:
        print(props)
    else:
        warnings.filterwarnings("ignore", category=FutureWarning)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    if debug:
        logger.info(sys.argv)
        logger.info(config)

    if model_name in [
        "BPR", "UUPretrain", "ReRec", "AllReRec", "IITest", "TestGames",
        "SparseReRec", "TestPantry", "TestOffice", "IITestDiag",
        "IITestDiagNew", "TestOfficeBPR", "TestOfficeUUPretrain",
        "UserReRec", "AgentCF", "RAAgentCF",
    ]:
        dataset = BPRDataset(config)
    elif model_name in ["UUTest", "UUTestDiag", "TestOfficeUUTest"]:
        dataset = ITEMBPRDataset(config)
    else:
        dataset = create_dataset(config)

    if debug:
        logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    if debug:
        logger.info(model)

    construct_transform(config)
    if model_name == "RAAgentCF":
        trainer = RAAgentCFTrainer(config, model, dataset)
    elif model_name in ["SASRec", "BPRMF"]:
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    else:
        trainer = LanguageLossTrainer(config, model, dataset)

    if not config["test_only"]:
        trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    test_result = trainer.evaluate(
        test_data,
        model_file="./AgentCF-Sep-07-2024_16-09-29.pth",
        load_best_model=False,
        show_progress=config["show_progress"],
    )
    print(test_result)
    if debug:
        logger.info(set_color("test result", "yellow") + f": {test_result}")
    return model_name, dataset_name, {"test_result": test_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="RAAgentCF", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="CDs-100-user-dense", help="name of datasets")
    parser.add_argument("--debug", nargs="?", const=True, default=None, type=parse_bool, help="print verbose debug output")
    args, _ = parser.parse_known_args()
    config_overrides = {}
    if args.debug is not None:
        config_overrides["debug"] = args.debug
    run_baseline(args.model, args.dataset, **config_overrides)
