# HilbertFlow 3D replay

Open `index.html` in Chrome, Edge or Firefox. It works offline; WebGL is required.
Drag either view to orbit both cameras, Shift-drag to pan, and scroll to zoom.
Use FPS sampling / Semantic segmentation to change modes. Play advances actual
saved or computed data, Next advances one round/stage, and the slider scrubs.
The cutaway clips upper points for room interiors; it never changes computation.

## FPS

The 93K, 202K and 503K scenes are real S3DIS inputs. Replay uses all input points,
10-bit Skilling Hilbert codes and fixed 256-point buckets. The viewer displays a
deterministic 24K-point subset plus the complete selected centroid sequence.
Every one of 256 distance-update rounds is checked against a dense FPS oracle.
Green means distance evaluated this round, grey means its partition was safely
skipped for this update, yellow is a selected sample and red is the current center.
Skipped points remain eligible in later rounds: pruning never deletes them.
Actual bucket AABBs are drawn when enabled.

The replay is a float64 reference of the fine-bucket bound in `csrc/fps_curve.cpp`,
generated locally. It is not a recorded RK3588 instruction trace. Runtime counters
on screen describe this replay. Latency and speedup use the original RK3588 CSVs
for N→4096 sampling, independently of the 256 replay rounds.

## Segmentation

The 93,133-point S3DIS scene supports PointNeXt-L and PointVector-XL saved
checkpoint predictions. The 202K and 503K scenes support PointNeXt-L, paired
with the fixed-clock paper figure measurements for those point counts.
RGB, geometry and output labels use the same point order.
Encoder views use saved sample0..3 index arrays; final labels are argmax of saved
13-class logits. Intermediate views illustrate the hierarchy, not reconstructed
feature tensors or a timing trace. The prediction artifacts are CUDA checkpoint
references, labelled in the interface. RK3588 timings are separate measured B0
and full-stack results for these models at this input size. Model-wide pruning
is not available in those timing CSVs and is displayed as N/A.

## Rebuild

Run `python3 demo/rk3588_fps/build_demo.py` inside the original VD4FPS workspace.
Requires NumPy, original scene files, index arrays and prediction artifacts.
The standalone HTML already embeds everything needed for replay. The ZIP also
contains the metric CSVs and a replay audit. No live inference service is invoked.
