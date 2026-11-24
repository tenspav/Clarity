"""Compare a Core ML output dump (FP16 IOSurface, stacked planes) with the PyTorch FP32 residual."""
import sys
import numpy as np
import torch
from torch.nn import functional as F
from model import CAP, HEADROOM, LumaResidual

ckpt, frame, dump, w, h = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
st = torch.load(ckpt, map_location="cpu", weights_only=True)
m = LumaResidual(st["kind"]).eval()
m.load_state_dict(st["model"])
x = np.fromfile(frame, "<u2").reshape(h, w).astype(np.float32)
t = torch.from_numpy(np.clip((x - 64) / 876, 0, 1))[None, None]
with torch.no_grad():
    ref = m.native(t)[0, 0].numpy() * HEADROOM
d = np.fromfile(dump, "<f2").astype(np.float32)
if st["kind"] == "Unpacked8":
    got = d.reshape(h, w)
else:
    planes = d.reshape(4, h // 2, w // 2)  # TL TR BL BR
    got = F.pixel_shuffle(torch.from_numpy(planes)[None], 2)[0, 0].numpy()
got *= HEADROOM
e = np.clip(got, -CAP, CAP) - np.clip(ref, -CAP, CAP)
print(f"{st['kind']} {w}x{h}: ref rms {np.sqrt((ref**2).mean()):.4f} code, coreml rms {np.sqrt((got**2).mean()):.4f}, "
      f"diff rms {np.sqrt((e**2).mean()):.4f} max {np.abs(e).max():.4f} code, signed min/max ref {ref.min():.3f}/{ref.max():.3f} "
      f"coreml {got.min():.3f}/{got.max():.3f}")
