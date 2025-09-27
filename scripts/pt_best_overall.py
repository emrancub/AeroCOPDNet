# ============================================================
# model_summary_from_pt.py — Keras-like summary from .pt file
# Prints: Layer (name/type), Output shape, Param #
# Also exports CSV. Handles DataParallel 'module.' prefixes.
# Includes Windows OpenMP duplicate DLL workaround.
# ============================================================

import os
# ---- Fix your earlier crash (OpenMP duplicate init on Windows) ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import importlib
from collections import OrderedDict
from typing import Any, Dict, Tuple, List, Optional, Union

import torch
import torch.nn as nn

# =============== USER CONFIG ===============
ckpt_path = r"C:\Users\HASAN MD EMRAN\PycharmProjects\NewCOPDProject\artifacts\cv\cvtest_20250908_182400_5f\best_overall.pt"

# Provide your model as "python.module.path:ClassName"
# Example: "models.copdnet:MyCOPDNet"
# If you don't know it or can't import, leave empty: OUTPUT SHAPES CANNOT BE DERIVED then.
MODEL_IMPORT = ""  # e.g., "models.copdnet:MyCOPDNet"
MODEL_KWARGS: Dict[str, Any] = {}  # constructor kwargs if any

# Dummy input size for forward pass (batch, channels, H, W) or (batch, features) etc.
# Adjust to what your model expects.
INPUT_SIZE: Tuple[int, ...] = (1, 3, 224, 224)
DEVICE = "cpu"
CSV_OUT = None  # or r"C:\path\to\summary.csv"
# ==========================================


# ---------- Utilities ----------
def strip_dataparallel_prefix(sd: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def find_state_dict(obj: Any) -> Optional["OrderedDict[str, torch.Tensor]"]:
    if isinstance(obj, (dict, OrderedDict)):
        # Common keys used by different trainers
        for key in ["state_dict", "model_state_dict", "model_state", "net", "model", "module"]:
            if key in obj and isinstance(obj[key], (dict, OrderedDict)):
                sub = obj[key]
                if all(isinstance(v, torch.Tensor) for v in sub.values()):
                    return OrderedDict(sub)
        # Or it might already be a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj)
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    return None

def load_checkpoint_any(path: str, map_location: str = "cpu") -> Any:
    # Try TorchScript first (if someone saved a scripted/traced module)
    try:
        ts = torch.jit.load(path, map_location=map_location)
        # TorchScript doesn't give easy per-layer output hooks, so we’ll still
        # prefer a Python nn.Module class if available.
        return {"_torchscript": ts, "_state_dict": ts.state_dict()}
    except Exception:
        pass
    # Fallback: regular torch.load
    return torch.load(path, map_location=map_location)

def human_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"

def count_params(module: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
    total = sum(p.numel() for p in module.parameters(recurse=False))
    return total, trainable


# ---------- Hook-based summarizer ----------
class LayerRecord:
    def __init__(self, name: str, m: nn.Module):
        self.name = name
        self.type = m.__class__.__name__
        self.out_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None
        self.param_total, self.param_trainable = count_params(m)

def is_leaf(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

def build_summary(model: nn.Module, example_input: torch.Tensor) -> List[LayerRecord]:
    model.eval()
    records: Dict[str, LayerRecord] = {}
    hooks = []

    # We collect only leaf layers to avoid duplications
    for name, m in model.named_modules():
        if name == "" or not is_leaf(m):
            continue
        records[name] = LayerRecord(name, m)

        def hook_factory(layer_name: str):
            def hook(module, inputs, outputs):
                # outputs might be Tensor, tuple, or list; capture shapes
                def shape_of(x):
                    if isinstance(x, torch.Tensor):
                        return tuple(x.shape)
                    return None
                if isinstance(outputs, (tuple, list)):
                    rec = [s for s in (shape_of(o) for o in outputs) if s is not None]
                    records[layer_name].out_shape = rec if rec else None
                else:
                    records[layer_name].out_shape = shape_of(outputs)
            return hook

        hooks.append(m.register_forward_hook(hook_factory(name)))

    with torch.no_grad():
        try:
            _ = model(example_input.to(next(model.parameters()).device if any(True for _ in model.parameters()) else DEVICE))
        except Exception as e:
            # Try CPU explicitly if device mismatch
            model.to(DEVICE)
            _ = model(example_input.to(DEVICE))

    for h in hooks:
        h.remove()

    # Preserve definition order as much as possible
    ordered = [records[k] for k in records.keys()]
    return ordered


# ---------- Optional dynamic import ----------
def instantiate_model(import_str: str, kwargs: Dict[str, Any]) -> nn.Module:
    """
    import_str = "package.module:ClassName"
    """
    module_name, class_name = import_str.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


# ---------- Main ----------
def main():
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: file not found → {ckpt_path}")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    obj = load_checkpoint_any(ckpt_path, map_location=DEVICE)

    # Determine state_dict
    if isinstance(obj, dict) and "_state_dict" in obj:  # TorchScript path
        state_dict = obj["_state_dict"]
    else:
        state_dict = find_state_dict(obj)
        if state_dict is None:
            print("[ERROR] Could not find a state_dict in this .pt file.")
            if isinstance(obj, dict):
                print("Available keys:", list(obj.keys()))
            sys.exit(2)

    state_dict = strip_dataparallel_prefix(OrderedDict(state_dict))

    # If we cannot import the model, we cannot run hooks to get Output shapes.
    if not MODEL_IMPORT:
        print("\n[WARN] MODEL_IMPORT not set. Will only report parameter tensors from state_dict (no Output shapes).")
        # Minimal report from state_dict
        rows = []
        total_params = 0
        total_bytes = 0
        for name, t in state_dict.items():
            if not torch.is_tensor(t):
                continue
            numel = t.numel()
            total_params += numel
            total_bytes += t.element_size() * numel
            rows.append((name, list(t.shape), str(t.dtype).replace("torch.", ""), numel))
        # Print a few
        print("\nName".ljust(70), "Shape".ljust(20), "DType".ljust(10), "Param #")
        print("-" * 115)
        for r in rows[:20]:
            n, s, d, k = r
            dn = (n[:67] + "...") if len(n) > 70 else n
            print(dn.ljust(70), str(s).ljust(20), d.ljust(10), f"{k:,}")
        if len(rows) > 20:
            print(f"... ({len(rows)-20} more)")

        print(f"\n[SUMMARY] Total parameters: {total_params:,}  |  Approx size: {human_bytes(total_bytes)}")
        if CSV_OUT:
            import csv
            with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["tensor_name", "shape", "dtype", "numel"])
                for n, s, d, k in rows:
                    w.writerow([n, s, d, k])
            print(f"[OK] Wrote CSV: {CSV_OUT}")
        print("\nTIP: To get Layer / Output shape / Param #, set MODEL_IMPORT='your.module:YourClass' and INPUT_SIZE properly.")
        return

    # Recreate model & load weights
    try:
        model = instantiate_model(MODEL_IMPORT, MODEL_KWARGS)
    except Exception as e:
        print(f"[ERROR] Failed to import/instantiate model from '{MODEL_IMPORT}': {e}")
        sys.exit(3)

    model.to(DEVICE)

    # Try strict, then fallback to non-strict with diagnostics
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Weights loaded (strict=True).")
    except Exception as e:
        print("[WARN] strict=True failed →", e)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[INFO] Loaded with strict=False.")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)

    # Build dummy input
    try:
        dummy = torch.zeros(*INPUT_SIZE)
    except Exception:
        print(f"[ERROR] Bad INPUT_SIZE={INPUT_SIZE}. Provide a tuple like (1,3,224,224) or (1,features).")
        sys.exit(4)

    # Hook-based layer summary
    records = build_summary(model, dummy)

    # Print table
    header = f"{'Layer (name/type)':50s} {'Output shape':30s} {'Param #':>12s}"
    print("\n" + header)
    print("-" * len(header))
    total_params = 0
    for rec in records:
        out_shape = rec.out_shape
        if isinstance(out_shape, list):
            osh = "[" + ", ".join(str(s) for s in out_shape) + "]"
        else:
            osh = str(out_shape) if out_shape is not None else "None"

        layer_disp = f"{rec.name} ({rec.type})"
        if len(layer_disp) > 50:
            layer_disp = layer_disp[:47] + "..."

        print(f"{layer_disp:50s} {osh:30s} {rec.param_total:12,d}")
        total_params += rec.param_total

    # Totals
    model_total = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * len(header))
    print(f"{'TOTAL (leaf modules sum)':50s} {'':30s} {total_params:12,}")
    print(f"{'TOTAL (model.parameters())':50s} {'':30s} {model_total:12,}  (trainable: {trainable_total:,})")

    # Optional CSV
    if CSV_OUT:
        import csv
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["layer_name", "layer_type", "output_shape", "param_num"])
            for rec in records:
                out_shape = rec.out_shape
                if isinstance(out_shape, list):
                    osh = "[" + ", ".join(str(s) for s in out_shape) + "]"
                else:
                    osh = str(out_shape) if out_shape is not None else ""
                w.writerow([rec.name, rec.type, osh, rec.param_total])
        print(f"[OK] Wrote CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
