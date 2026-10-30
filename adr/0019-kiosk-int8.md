# 0019. On the kiosk, run the final model quantized to int8

- **Status:** Accepted (complements [0012](0012-final-model.md))
- **Date:** 2026-10-26

**Context.** The kiosks run on modest CPUs. P09 timed the final ResNet-18 at 70 ms per photo on one thread of another computer.
`vision/P16_kiosk_cost.ipynb` tries two ways to make it cheaper without a weaker model, with rules written before the runs:
post-training int8 quantization of the final model (accepted if within 1 point of float32 on validation), and MobileNetV3,
networks designed for phones, frozen with a new final layer (5 seeds; scored on museum B if within 2 points of the final model).
Every network was re-timed on one computer (ARM, macOS) with the P09 protocol. Measured (`results/kiosk_cost.csv`,
`results/experiments.csv`):

| candidate | validation | museum B | 1 thread | 4 threads | size |
|---|---|---|---|---|---|
| final ResNet-18, float32 (ADR 0012) | 83.3% | 86.0% | 16.6 ms | 11.0 ms | 44.8 MB |
| **final ResNet-18, int8** | **83.3%** | **86.2%** | **12.3 ms** | 13.2 ms | **11.2 MB** |
| MobileNetV3-Large, frozen | 82.9% (5 seeds) | 84.8% | 19.1 ms | 44.4 ms | 12.1 MB |
| MobileNetV3-Small, frozen | 80.2% (5 seeds) | not scored | 11.2 ms | 32.3 ms | 3.8 MB |

**Decision.** The kiosk runs the final model of ADR 0012 quantized to int8 (static post-training quantization, PyTorch FX,
calibrated on 200 training paintings; `fbgemm` on an x86 kiosk, `qnnpack` on ARM), on one thread. The float32 model stays the
reference of the project and of the demo.

**Rejected.** MobileNetV3-Large (as accurate within the noise and as small, but slower on this CPU and much slower on several
threads); MobileNetV3-Small (3 points lower); ResNet-50 (1.6 times the time, ADR 0012).

**Consequences.**
- Same accuracy on both sets (the int8 and float32 models agree on 98.1% of the validation paintings), 4 times smaller, 26%
  faster on one thread. On 4 threads int8 is slower than float32: one thread per photo.
- **Operations do not predict time.** MobileNetV3-Large needs about 8 times fewer operations than ResNet-18 (published figures)
  and is slower here: its depthwise convolutions are poorly optimised in PyTorch on this CPU. The kiosk's own processor must be
  measured before the choice is final; ADR 0012 and this ADR only fix the ranking measured here.
- This computer is about 4 times faster than P09's (16.6 against 70 ms for the same network); the ratios between networks agree
  (small CNN 2.5% against 2.4% of the ResNet-18; ResNet-50 1.6 against 2.0 times).
- The int8 model is rebuilt by P16 (seeded calibration) and not saved; the demo does not use it.
