"""Export fixed Core ML graphs. On Linux, conversion only, no Core ML runtime."""
import argparse
import platform
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from model import LumaResidual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ios18",action="store_true",help="export with an iOS 18 deployment target")
    args = p.parse_args()
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    kind = state["kind"]
    model = LumaResidual(kind).eval()
    model.load_state_dict(state["model"],strict=True)
    args.out.mkdir(parents=True,exist_ok=True)
    for h,w in [(1080,1920),(720,1280)]:
        shape = (1,1,h,w) if kind == "Unpacked8" else (1,4,h//2,w//2)
        traced = torch.jit.trace(model,torch.zeros(shape),check_trace=True).eval()
        for image in ([False,True] if kind=="Unpacked8" else [False]):
            if image:
                args_io = {
                    "inputs":[ct.ImageType(name="luma",shape=shape,
                               color_layout=ct.colorlayout.GRAYSCALE_FLOAT16,scale=1,bias=0)],
                    "outputs":[ct.ImageType(name="delta",color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                }
            else:
                args_io = {"inputs":[ct.TensorType(name="luma",shape=shape,dtype=np.float16)],
                           "outputs":[ct.TensorType(name="delta",dtype=np.float16)]}
            m = ct.convert(traced,convert_to="mlprogram",compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18 if args.ios18 else ct.target.iOS16,
                           compute_units=ct.ComputeUnit.CPU_AND_NE,
                           skip_model_load=platform.system()!="Darwin", **args_io)
            m.short_description = "Signed residual only. Input nominal HDR luma (code10-64)/876."
            m.input_description["luma"] = "FP16 Y prime, normalized narrow range. Packed channels: TL TR BL BR."
            m.output_description["delta"] = "Signed normalized residual. Clamp and apply in the renderer."
            out = args.out/f"ClarityLuma1x-{kind}-{h}{'-image' if image else ''}.mlpackage"
            if out.exists():
                raise FileExistsError(out)
            m.save(str(out))
            print(out)


if __name__ == "__main__":
    main()
