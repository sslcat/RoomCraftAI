"""Approved-Pod-only full pinned Qwen loading probe; no training or image generation."""

import argparse
import time
from pathlib import Path

import torch

from roomcraft.common import write_json
from roomcraft.editing import EditConfig, edit_preflight, load_edit_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be new")
    preflight = edit_preflight()
    if not preflight["can_attempt"]:
        write_json(args.output, preflight)
        raise RuntimeError("failed GPU/RAM/environment gate before weight download")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pipe = load_edit_pipeline(EditConfig(size=512))
    torch.cuda.synchronize()
    assert pipe.transformer.config.zero_cond_t
    write_json(
        args.output,
        {
            "status": "full_pretrained_pipeline_loaded_no_optimizer_updates",
            "preflight": preflight,
            "seconds_including_download": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_free_total_bytes": list(torch.cuda.mem_get_info()),
            "limitation": "Loading fit does not prove forward/backward activation fit",
        },
    )


if __name__ == "__main__":
    main()
