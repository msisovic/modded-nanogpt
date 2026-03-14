"""Check if torch.compile fuses RMSNorm + FP8 quantization."""
import torch
import torch.nn.functional as F
import os

os.environ["TORCH_LOGS"] = "+output_code"

device = "cuda"
M, K = 49152, 768

x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

# Pattern 1: What we currently do (dynamic scale)
def norm_then_quant_dynamic(x):
    normed = F.rms_norm(x, (x.size(-1),))
    amax = normed.detach().abs().max().clamp(min=1e-12)
    x_f8 = (normed.detach() * (448.0 / amax)).to(torch.float8_e4m3fn)
    scale = amax / 448.0
    return normed, x_f8, scale

# Pattern 2: Fixed scale (no amax needed)
def norm_then_quant_fixed(x):
    normed = F.rms_norm(x, (x.size(-1),))
    x_f8 = (normed * 28.0).to(torch.float8_e4m3fn)  # fixed scale_inv = 448/16
    return normed, x_f8

print("=" * 60)
print("Compiling dynamic scale pattern...")
print("=" * 60)
compiled_dynamic = torch.compile(norm_then_quant_dynamic, dynamic=False, fullgraph=True)
try:
    out = compiled_dynamic(x)
    print(f"Dynamic: compiled OK, outputs: {[o.shape for o in out]}")
except Exception as e:
    print(f"Dynamic: FAILED - {e}")

print("\n" + "=" * 60)
print("Compiling fixed scale pattern...")
print("=" * 60)
compiled_fixed = torch.compile(norm_then_quant_fixed, dynamic=False, fullgraph=True)
try:
    out = compiled_fixed(x)
    print(f"Fixed: compiled OK, outputs: {[o.shape for o in out]}")
except Exception as e:
    print(f"Fixed: FAILED - {e}")
