# Kefirstan VTuber 2.2

- Removed the separate white facePatch responsible for the 3D nose protrusion.
- Direct head pose every OpenGL frame, no exponential smoothing.
- 480x360 ML Kit FAST face detection; latest-frame backpressure and no contours.
- Preserved landscape rotation for 16:9.

The camera and ML Kit still have nonzero physical processing latency. The GitHub build is not a substitute for a device performance test.
