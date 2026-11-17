"""python -m planar_arm.pc02 --output runs/pc02_seed42"""

import argparse
from dataclasses import asdict

from .experiment import run
from .inference import InferenceConfig


def main():
    parser = argparse.ArgumentParser(description="PC-02: inferencia PC y referencia de optimización de entradas")
    parser.add_argument("--checkpoint", default="runs/pc01_seed42/model.pt")
    parser.add_argument("--output", default="runs/pc02_seed42")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--initial", type=float, nargs=2, action="append", help="Inicialización explícita (repetible)")
    parser.add_argument("--target", type=float, nargs=2, action="append", help="Objetivo (repetible); por defecto cuatro objetivos")
    for name, default in asdict(InferenceConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args())
    checkpoint, output = args.pop("checkpoint"), args.pop("output")
    starts, initials, targets = args.pop("starts"), args.pop("initial"), args.pop("target")
    try:
        run(checkpoint, output, InferenceConfig(**args), targets, starts, initials)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
