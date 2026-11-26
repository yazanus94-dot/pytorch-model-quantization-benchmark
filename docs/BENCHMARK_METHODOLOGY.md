# Benchmark Methodology

- Same pretrained ResNet18 weights and preprocessing for all variants.
- Same validation slice, batch size, CPU thread count, warmup, iterations.
- Timing after warmup under `torch.inference_mode()`.
- Serialized size = `torch.save(state_dict())` bytes.
- Peak RSS = max process resident memory during timed inference (process-wide).
- Accuracy on Imagenette validation slice remapped to ImageNet-1K indices.
