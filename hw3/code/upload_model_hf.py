import argparse
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser("Upload model to HuggingFace Hub")
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


model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.push_to_hub(
    repo_id=args.model_name, private=True, token="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"
)
