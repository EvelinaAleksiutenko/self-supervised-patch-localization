from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from src.config.config import Config
from src.utils.data import source_transform, patch_transform
from src.utils.losses import ncc_loss
from src.utils.metrics import compute_metrics
from src.utils.model import SiamesePatchLocalizer, extract_region

log = logging.getLogger(__name__)



def _load_dataset(dataset_dir: str, cfg: Config):
    """Load source images, patches, and GT coords from the provided dataset directory.

    Expected format:
        dataset_dir/
            coords.csv          (columns: index, y_start, x_start)
            source/{index:05d}.png
            patch/{index:05d}.png
    """
    root = Path(dataset_dir)
    csv_path = root / "coords.csv"

    img_tf = source_transform(cfg)
    patch_tf = patch_transform(cfg)

    sources, patches, gts = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["index"])
            fname = f"{idx:05d}.png"
            src_img = Image.open(root / "source" / fname).convert("RGB")
            pat_img = Image.open(root / "patch" / fname).convert("RGB")
            sources.append(img_tf(src_img))
            patches.append(patch_tf(pat_img))
            gts.append(torch.tensor([float(row["y_start"]), float(row["x_start"])]))

    return (
        torch.stack(sources),
        torch.stack(patches),
        torch.stack(gts),
    )



@torch.no_grad()
def evaluate(cfg: Config, checkpoint_path: str, dataset_dir: str, batch_size: int) -> dict[str, float]:
    device = torch.device(cfg.device)

    # Load model from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    model = SiamesePatchLocalizer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint: %s  (epoch %d, val_loss %.5f)",
             checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    # Load dataset
    sources, patches, gts = _load_dataset(dataset_dir, cfg)
    dataset = TensorDataset(sources, patches, gts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    log.info("Samples: %d", len(dataset))

    # Run evaluation
    total_ncc = 0.0
    all_preds: list[torch.Tensor] = []
    all_gts: list[torch.Tensor] = []
    total_inference_time = 0.0
    n_samples = 0

    for src_batch, pat_batch, gt_batch in loader:
        src_batch = src_batch.to(device)
        pat_batch = pat_batch.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        coords, _ = model(src_batch, pat_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_inference_time += t1 - t0
        n_samples += src_batch.size(0)

        region = extract_region(src_batch, coords, cfg)
        total_ncc += ncc_loss(region, pat_batch).item() * src_batch.size(0)
        all_preds.append(coords.cpu())
        all_gts.append(gt_batch)

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_gts))
    metrics["loss"] = total_ncc / len(dataset)
    metrics["total_inference_time_s"] = total_inference_time
    metrics["avg_inference_time_ms"] = (total_inference_time / n_samples) * 1000
    metrics["throughput_samples_per_s"] = n_samples / total_inference_time

    return metrics



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate patch localizer on a dataset")
    parser.add_argument("checkpoint", type=str,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("dataset", type=str,
                        help="Path to dataset directory (coords.csv + source/ + patch/)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device

    metrics = evaluate(cfg, args.checkpoint, args.dataset, args.batch_size)

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  NCC Loss          : {metrics['loss']:.5f}")
    print(f"  Mean ED           : {metrics['mean_ed']:.3f} px")
    print(f"  Median ED         : {metrics['median_ed']:.3f} px")
    print(f"  MAE y             : {metrics['mae_y']:.3f} px")
    print(f"  MAE x             : {metrics['mae_x']:.3f} px")
    print(f"  Acc@1px           : {metrics['acc_at_1']:.4f}")
    print(f"  Acc@2px           : {metrics['acc_at_2']:.4f}")
    print(f"  Acc@5px           : {metrics['acc_at_5']:.4f}")
    print("-" * 50)
    print(f"  Avg inference     : {metrics['avg_inference_time_ms']:.2f} ms/sample")
    print(f"  Throughput        : {metrics['throughput_samples_per_s']:.1f} samples/s")
    print(f"  Total inference   : {metrics['total_inference_time_s']:.3f} s")
    print("=" * 50)

    print(f"\nMean Euclidean Distance: {metrics['mean_ed']:.3f} px")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
