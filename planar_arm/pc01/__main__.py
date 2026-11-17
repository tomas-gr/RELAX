"""Entrenar: python -m planar_arm.pc01 --output runs/pc01_seed42."""

import argparse
from dataclasses import asdict
import json
import torch

from planar_arm import PlanarArm
from planar_arm.data import generate_dataset
from .evaluation import evaluate
from .experiment import Config, run
from .model import load_model


def main():
    parser = argparse.ArgumentParser(description="PC-01: entrenamiento PC de cinemática directa")
    parser.add_argument("--output", default="runs/pc01_seed42")
    parser.add_argument("--load", help="Evaluar un checkpoint sin entrenar; usa su configuración guardada")
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args())
    output, load = args.pop("output"), args.pop("load")
    try:
        if load:
            model, checkpoint = load_model(load)
            c = Config(**checkpoint["config"])
            torch.set_num_threads(c.threads)
            torch.use_deterministic_algorithms(True)
            data = generate_dataset(c.samples, c.seed, PlanarArm(c.l1, c.l2))
            report, _ = evaluate(model, data.test, (c.l1, c.l2))
            print(json.dumps(report, indent=2))
        else:
            run(Config(**args), output)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
