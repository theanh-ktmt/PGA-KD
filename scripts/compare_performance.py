#!/usr/bin/env python3
"""
Benchmark script for Multimodal Embedding Distillation
FIXED: Uses encode_input to prevent silent bypasses, accurate latency, isolated activation VRAM.
"""

import sys
import os
import torch
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import gc

# Ensure the script can find your 'src' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arguments import ModelArguments
from src.model.processor import (
    load_processor, 
    process_vlm_inputs_fns, 
    MODEL2BACKBONE,
    VLM_IMAGE_TOKENS,
    QWEN2_VL,
    QWEN2_5_VL,
    LLAVA_QWEN2,
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def create_dummy_image(target_size: tuple = (336, 336)) -> Image.Image:
    return Image.fromarray(
        np.random.randint(0, 255, (*target_size, 3), dtype=np.uint8)
    ).convert("RGB")


def format_text_with_image_token(text: str, backbone: str) -> str:
    image_token = VLM_IMAGE_TOKENS.get(backbone, "<image>")
    if backbone in [QWEN2_VL, QWEN2_5_VL, LLAVA_QWEN2]:
        return f"{image_token}\n{text}"
    return f"{image_token} {text}"


def create_model_args_for_benchmark(
    model_name: str,
    backbone: str,
    hidden_dim: int,
    checkpoint_path: Optional[str] = None,
) -> ModelArguments:
    return ModelArguments(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        model_backbone=backbone,
        student_hidden_dim=hidden_dim,
        teacher_hidden_dim=hidden_dim,
        lora=False,
        pooling="last",
        normalize=True,
        processor_name=None,
    )


@torch.no_grad()
def prepare_benchmark_inputs(
    processor,
    process_fn,
    backbone: str,
    device: str = "cuda",
    text: str = "Describe this image in great detail to test the embedding generation.",
    image_size: tuple = (336, 336),
) -> Dict[str, torch.Tensor]:
    dummy_image = create_dummy_image(image_size)
    formatted_text = format_text_with_image_token(text, backbone)
    
    model_inputs = {"text": [formatted_text], "images": [dummy_image]}
    
    try:
        processed = process_fn(model_inputs, processor=processor, max_length=128)
    except TypeError:
        processed = process_fn(model_inputs, processor=processor)
    
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                processed[k] = v.to(device, dtype=torch.bfloat16) # Align with model's bfloat16
            else:
                processed[k] = v.to(device)
                
    return processed


@torch.no_grad()
def run_model_forward(model: torch.nn.Module, inputs: Dict) -> Optional[torch.Tensor]:
    """
    Directly targets the MMEBModel's encode_input method to accurately 
    measure the forward pass generation of embeddings.
    """
    if hasattr(model, "encode_input"):
        outputs = model.encode_input(inputs)
        
        # Depending on the backbone, encode_input returns a single tensor or a tuple
        if isinstance(outputs, tuple):
            return outputs[0]  # This is usually the pooled_output
        return outputs
    else:
        # Fallback just in case a raw base_model is passed instead of MMEBModel
        kwargs = dict(inputs)
        outputs = model(**kwargs)
        return outputs


def estimate_flops_manual(model: torch.nn.Module, inputs: Dict, latency_sec: float) -> float:
    params = sum(p.numel() for p in model.parameters())
    
    if 'input_ids' in inputs and inputs['input_ids'] is not None:
        seq_len = inputs['input_ids'].shape[1] if len(inputs['input_ids'].shape) > 1 else 128
    else:
        seq_len = 128
        
    estimated_flops = 2 * params * seq_len
    
    if latency_sec > 0:
        tflops = (estimated_flops / 1e12) / latency_sec
        return tflops, True
    return float('nan'), False


@torch.no_grad()
def benchmark_forward_pass(
    model: torch.nn.Module,
    prepare_fn,
    model_name: str,
    device: str = "cuda",
    warmup: int = 5,
    runs: int = 20,
    batch_multiplier: int = 2,
) -> Dict[str, float]:
    
    print(f"🔍 Benchmarking: {model_name}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    inputs = prepare_fn(device)
    
    print("   -> Running warmup...")
    # Warmup with synchronization
    for _ in range(warmup):
        _ = run_model_forward(model, inputs)
        torch.cuda.synchronize()
    
    # Establish base VRAM (Weights only) before timed forward passes
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    
    print("   -> Running timed passes...")
    start_time = time.perf_counter()
    
    for _ in range(runs):
        for _ in range(batch_multiplier):
            _ = run_model_forward(model, inputs)
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    total_forwards = runs * batch_multiplier
    total_time_sec = end_time - start_time
    avg_latency_ms = (total_time_sec / total_forwards) * 1000
    
    # Peak memory - Base Memory = Memory used purely by Activations (Forward Pass)
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    act_vram_gb = peak_memory_gb - base_memory_gb
    
    tflops, flops_available = estimate_flops_manual(model, inputs, avg_latency_ms / 1000)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    return {
        "model": model_name.split("/")[-1],
        "params_m": params_m,
        "base_vram_gb": base_memory_gb,
        "act_vram_gb": act_vram_gb,
        "latency_ms": avg_latency_ms,
        "tflops": tflops,
        "flops_available": flops_available,
        "total_forwards_measured": total_forwards,
    }


def load_model_for_benchmark(
    model_name: str,
    backbone: str,
    hidden_dim: int,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
):
    print(f"\n⏳ Loading {model_name} (backbone: {backbone})...")
    
    model_args = create_model_args_for_benchmark(
        model_name=model_name,
        backbone=backbone,
        hidden_dim=hidden_dim,
        checkpoint_path=checkpoint_path,
    )
    
    processor = load_processor(model_args, data_args=None)
    
    from src.model.model import MMEBModel
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    
    # Push to device. MMEBModel uses bfloat16 heavily in its code.
    model.to(device, dtype=torch.bfloat16)
    
    process_fn = process_vlm_inputs_fns.get(backbone)
    if process_fn is None:
        raise ValueError(f"No process_fn for backbone: {backbone}")
        
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Loaded: {model_name} ({params_m:.1f}M params)")
    return model, processor, process_fn


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    assert DEVICE == "cuda", "GPU required for accurate benchmarking."
    
    print(f"🚀 Benchmark on {DEVICE}")
    print("=" * 125)
    
    TEACHER_CONFIG = {
        "name": "raghavlite/B3_Qwen2_2B",
        "backbone": "qwen2_vl",
        "hidden_dim": 1536,
        "checkpoint_path": None,
    }
    
    STUDENT_CONFIG = {
        "name": "apple/FastVLM-0.5B",
        "backbone": "llava_qwen2",
        "hidden_dim": 896,
        "checkpoint_path": None,
    }
    
    results = []
    
    # ================= TEACHER =================
    teacher_model, teacher_proc, teacher_fn = load_model_for_benchmark(
        model_name=TEACHER_CONFIG["name"],
        backbone=TEACHER_CONFIG["backbone"],
        hidden_dim=TEACHER_CONFIG["hidden_dim"],
        device=DEVICE,
        checkpoint_path=TEACHER_CONFIG["checkpoint_path"],
    )
    
    def teacher_prepare_fn(device):
        return prepare_benchmark_inputs(
            processor=teacher_proc,
            process_fn=teacher_fn,
            backbone=TEACHER_CONFIG["backbone"],
            device=device,
        )
        
    teacher_result = benchmark_forward_pass(
        model=teacher_model,
        prepare_fn=teacher_prepare_fn,
        model_name=TEACHER_CONFIG["name"],
        device=DEVICE,
        runs=10, 
        batch_multiplier=2,
    )
    teacher_result["note"] = "Matryoshka: output dim only, compute unchanged"
    results.append(teacher_result)
    
    # Free up memory aggressively
    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    
    # ================= STUDENT =================
    student_model, student_proc, student_fn = load_model_for_benchmark(
        model_name=STUDENT_CONFIG["name"],
        backbone=STUDENT_CONFIG["backbone"],
        hidden_dim=STUDENT_CONFIG["hidden_dim"],
        device=DEVICE,
        checkpoint_path=STUDENT_CONFIG["checkpoint_path"],
    )
    
    def student_prepare_fn(device):
        return prepare_benchmark_inputs(
            processor=student_proc,
            process_fn=student_fn,
            backbone=STUDENT_CONFIG["backbone"],
            device=device,
        )
        
    student_result = benchmark_forward_pass(
        model=student_model,
        prepare_fn=student_prepare_fn,
        model_name=STUDENT_CONFIG["name"],
        device=DEVICE,
        runs=10,
        batch_multiplier=2,
    )
    student_result["note"] = "Distilled student - reduced params/compute"
    results.append(student_result)
    
    # ================= OUTPUT =================
    print("\n" + "=" * 125)
    print("📊 BENCHMARK RESULTS - Multimodal Embedding Distillation (Forward Pass / Encoding)")
    print("=" * 125)
    print(f"{'Model':<20} {'Params(M)':<12} {'Weight VRAM':<14} {'Act VRAM':<12} {'Latency(ms)':<15} {'TFLOPS':<10} {'Notes'}")
    print("-" * 125)
    
    for r in results:
        latency_str = f"{r['latency_ms']:.2f}"
        tflops_str = f"{r['tflops']:.2f}" if r['flops_available'] else "est."
        weight_vram = f"{r['base_vram_gb']:.2f} GB"
        act_vram = f"{r['act_vram_gb']:.2f} GB"
        
        print(
            f"{r['model']:<20} "
            f"{r['params_m']:<12.1f} "
            f"{weight_vram:<14} "
            f"{act_vram:<12} "
            f"{latency_str:<15} "
            f"{tflops_str:<10} "
            f"{r['note']}"
        )
        
    print("=" * 125)
    
    if len(results) == 2:
        t, s = results[0], results[1]
        if s["latency_ms"] > 0 and t["latency_ms"] > 0:
            speedup = t["latency_ms"] / s["latency_ms"]
            vram_savings = t["act_vram_gb"] / s["act_vram_gb"] if s["act_vram_gb"] > 0 else 0
            
            print(f"\n💡 Student vs Teacher:")
            print(f"   • Latency Speedup: ~{speedup:.2f}x faster inference speed")
            if vram_savings > 0:
                print(f"   • VRAM Efficiency: Uses ~{vram_savings:.2f}x less Activation VRAM during generation")


if __name__ == "__main__":
    main()