import argparse
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser("Download model from HuggingFace Hub")

parser.add_argument(
    "--model-dir",
    "-d",
    type=str,
    default="./out",
    help="Directory where model is saved",
)
parser.add_argument(
    "--model-name",
    "-n",
    type=str,
    help="Name of the model to be uploaded",
    required=True,
)

args = parser.parse_args()


model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, cache_dir=args.model_dir
)
