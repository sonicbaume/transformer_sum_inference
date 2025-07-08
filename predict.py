from .extractive import ExtractiveSummarizer
import argparse
import json
from pathlib import Path

def predict(
    sentences: list[str],
    ratio: float,
    model_dir: str,
    checkpoint_file: str,
):
    hparams={
        "model_name_or_path": model_dir,
        "no_use_token_type_ids": True,
        "num_frozen_steps": 0,
        "pooling_mode": None,
        "classifier_dropout": None,
        "tokenizer_name": None,
    }
    model = ExtractiveSummarizer.load_from_checkpoint(f"{model_dir}/{checkpoint_file}", hparams=hparams)
    model.eval()
    
    return model.predict_sentences(
        sentences,
        num_summary_sentences=round(len(sentences) * ratio),
        # return_ids=False # impo
        # raw_scores=True, # could feed these raw scores into an llm for refinement? 
    )

def parse_cli(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Defaults to `sys.argv[1:]`. 

    Returns
    -------
    Namespace
        Parsed arguments with attributes: sentences, ratio, checkpoint.
    """
    parser = argparse.ArgumentParser(
        description="Run sentence-level predictions with a trained checkpoint."
    )
    parser.add_argument(
        "-s",
        "--sentences",
        required=True,
        type=Path, 
        metavar="PATH",
        help="Path to sentences JSON file",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        required=True,
        type=float,
        metavar="FLOAT",
        help="Threshold/ratio in the range 0-1",
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        required=True,
        type=str,
        metavar="CKPT",
        help="Path to transformer model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "-c",
        "--model-checkpoint",
        required=False,
        default="facebook/bart-large-cnn",
        type=str,
        metavar="CKPT",
        help="Path to bart model checkpoint (.ckpt)",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.ratio < 1.0):
        parser.error("The ratio value must be between 0 and 1.")

    return args

def main(argv=None):
    try:
        args    = parse_cli(argv)
        results = predict(
            json.load(open(args.sentences, "r")), 
            args.ratio, 
            args.model_dir, 
            args.model_checkpoint
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except BaseException as e:
        print(f"predict error: {e}")

if __name__ == "__main__":
    main()
    