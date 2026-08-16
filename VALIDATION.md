# Validation Notes

## What is validated

- FP32, dynamic INT8, and static INT8 models all execute on CPU.
- Static post-training quantization is calibrated on a deterministic sample of Imagenette training images.
- Accuracy uses Imagenette validation labels remapped to the original ImageNet-1K output indices.
- Top-1 and top-5 counting has unit coverage.
- `RESULTS.md` is generated from raw benchmark objects rather than hand-edited numbers.
- Profiler traces use `torch.profiler` with CPU activity, shape recording, and memory profiling enabled.

## Fair-comparison controls

- Same pretrained ResNet18 weights and preprocessing are used for each model variant.
- Same validation slice, batch size, CPU thread count, warmup count, and timed iteration count are used across variants.
- Timing is performed after warmup with `torch.inference_mode()`.
- Results include the machine/software metadata needed to avoid comparing unlike hardware as if it were a model-only effect.

## Known limitations

1. Imagenette is a 10-class subset of ImageNet, so its measured accuracy is not the official ImageNet-1K validation accuracy.
2. Dynamic quantization has limited leverage on ResNet18 because supported dynamic quantization targets modules such as `Linear`, while convolution dominates ResNet18 compute.
3. Peak RSS is process-wide memory, not isolated tensor memory.
4. Eager `torch.ao.quantization` is a legacy API family. PyTorch is moving active quantization development toward torchao/PT2E; this repository deliberately keeps the eager path because it makes the static-calibration mechanics easy to inspect and directly demonstrates the requested API family.
