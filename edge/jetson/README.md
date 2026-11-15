# Jetson edge runner (stub)

`runner.py` is the bus-box version of the edge loop in `edge/protocol.md`: camera → distance-triggered keyframes →
segmenter (ONNX Runtime, or Ultralytics `.pt`) → IoU tracker → observations → upload (presign, observations, 1 Hz
fixes, trip close). It reuses the simulator's payload builders and HTTP client (`edge/sim/payloads.py`,
`edge/sim/client.py`), so the wire format is the same one the API tests use.

## Run it without a Jetson
Any video with a GPS track works (CPU, `onnxruntime`):
```bash
python -m edge.jetson.runner --video drive.mp4 --gps-csv drive_gps.csv --onnx weights/model.onnx --dry-run --out out/
python -m edge.jetson.runner --video drive.mp4 --gps-csv drive_gps.csv --onnx weights/model.onnx \
    --api http://localhost:8000 --device jetson-dev-01 --calib sim_bus
```
GPS CSV: `t_s,lat,lon[,speed_mps,heading_deg,accuracy_m]` (seconds from the first video frame) or `ts,...` with ISO
times. Speed and heading are derived from consecutive fixes when missing.

## On the bus (to do on real hardware: not done or tested here)
| Item | Plan |
|---|---|
| OS / runtime | JetPack 6.x (Orin Nano 8 GB), `onnxruntime-gpu` from the Jetson Zoo, or a TensorRT engine built with `trtexec --onnx=model.onnx --fp16` |
| Camera | IMX477/IMX219 CSI via `nvarguscamerasrc` (pipeline in `runner.py`'s docstring), or a USB UVC camera (`--video 0`) |
| GNSS | u-blox M8/M9 over USB; an NMEA reader (pyserial + pynmea2) writing the CSV format above, or replace `GpsTrack` with a live reader |
| IMU jolts | MPU-6050/BMI088 on I²C; emit `source: jolt` observations (see `edge/sim/bus.py` for the payload) |
| Privacy | Face/plate blur before `cv2.imencode` (`edge/sim/frames.py:Redactor`, YuNet ONNX) |
| Power | 24 V bus supply → isolated DC-DC 12 V/5 A, ignition-sense delayed shutdown (see `docs/DEPLOYMENT_TSRTC.md`) |
| Service | systemd unit, restart on failure, outbox on disk so a reboot does not lose observations |
| Upload | LTE en route or depot Wi-Fi (`upload_wifi_only` in the signed edge config) |

Calibration (camera height, pitch, intrinsics) follows `docs/CALIBRATION_SOP.md`; until a certificate is
`certified=true`, its measurements are prioritisation-grade only.
