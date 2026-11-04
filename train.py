import argparse
import hashlib
from multiprocessing import freeze_support
from pathlib import Path

import torch
import yaml

from torch import nn
from ultralytics import YOLO
from ultralytics.nn import modules, tasks
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import init_seeds


from modules.DGFM import DGFM
from modules.DSDM import DSDM
from modules.FAGM import FAGM

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "configs/FSDG-YOLO.yaml"
_standard_parser = tasks.parse_model


def parse_model(config, ch, verbose=True):
    definitions = config["backbone"] + config["head"]
    if not any(name in {"DSDM", "DGFM", "FAGM"} for _, _, name, _ in definitions):
        return _standard_parser(config, ch, verbose)

    scale = config.get("scale") or next(iter(config["scales"]))
    depth, width, limit = config["scales"][scale]
    blocks = {"Conv": modules.Conv, "DSDM": DSDM, "SPPF": modules.SPPF,
              "C3k2": modules.C3k2, "C2PSA": modules.C2PSA}
    layers, channels, saved = [], [], []
    for index, (source, repeats, name, args) in enumerate(definitions):
        sources = [source] if isinstance(source, int) else source
        inputs = [channels[i] if channels else ch for i in sources]
        out = inputs[0]
        if name == "nn.Upsample":
            size, factor, mode = args
            layer = nn.Upsample(None if size == "None" else size, factor, mode)
        elif name == "Concat":
            out = sum(inputs)
            layer = modules.Concat(*args)
        elif name == "DGFM":
            layer = DGFM(*inputs, *args)
        elif name == "FAGM":
            FAGM.legacy = False  # The YOLO11 backbone uses C3k2.
            layer = FAGM(config["nc"], *args[1:], reg_max=config.get("reg_max", 16),
                         end2end=config.get("end2end", False), ch=tuple(inputs))
        else:
            out = make_divisible(min(args[0], limit) * width, 8)
            options = list(args[1:])
            if name in {"C3k2", "C2PSA"}:
                if name == "C3k2" and scale in {"m", "l", "x"}:
                    options[0] = True
                options.insert(0, max(round(repeats * depth), 1))
            layer = blocks[name](inputs[0], out, *options)
        layer.i, layer.f = index, source
        layer.type = f"{type(layer).__module__}.{type(layer).__name__}"
        layer.np = sum(p.numel() for p in layer.parameters())
        if verbose:
            LOGGER.info(f"{index:>3} {str(source):>16} {layer.np:>10}  {name}")
        layers.append(layer)
        channels.append(out)
        saved.extend(i % index for i in sources if i != -1)
    return nn.Sequential(*layers), sorted(saved)


# Ultralytics also rebuilds the graph when the trainer loads its configuration.
tasks.parse_model = parse_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fasdd", "dfire"], default="fasdd")
    parser.add_argument("--data", type=Path, help="Path to the dataset YAML.")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--init", type=Path, default=ROOT / "weights/yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--check", action="store_true", help="Check the model on CPU and exit.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = yaml.safe_load((ROOT / f"configs/train_{args.dataset}.yaml").read_text("utf-8"))
    # The original seed-15 FASDD script first set the seed inside the trainer.
    if args.dataset == "dfire" or args.seed != 15:
        init_seeds(args.seed, deterministic=config["deterministic"])
    model = YOLO(str(MODEL), task="detect")
    if args.check:
        with torch.inference_mode():
            prediction = model.model.cpu().eval()(torch.zeros(1, 3, 320, 320))[0]
        if prediction.shape != (1, 6, 8500) or not torch.isfinite(prediction).all():
            raise RuntimeError("Unexpected detector output.")
        print("CPU forward OK. Detection strides:", model.model.stride.tolist())
        return
    if args.data is None or not args.data.is_file():
        parser.error("Pass an existing dataset YAML with --data.")
    data = args.data.resolve()
    if args.dataset == "fasdd":
        if not args.init.is_file() or hashlib.sha256(args.init.read_bytes()).hexdigest() != "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1":
            parser.error("Provide the original YOLO11n initialization file using --init.")
        model.load(str(args.init))
    config.update(data=str(data), seed=args.seed, device=args.device, project=str(ROOT / "runs"),
                  name=f"{args.dataset}_seed{args.seed}", exist_ok=False)
    model.train(**config)


if __name__ == "__main__":
    freeze_support()
    main()
