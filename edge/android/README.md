# RoadFix Edge — Android skeleton

Kotlin app for a phone mounted behind a TGSRTC bus windscreen. It implements the edge loop of `edge/protocol.md`:
CameraX analysis, 1 Hz fused-location trip fixes with `cam_active`, the distance keyframe trigger
(`KeyframeTrigger.kt`, same rule as `edge/sim/bus.py`), segmenter → observations in a durable outbox
(`Outbox.kt`), and a WorkManager upload in protocol order (presign + PUT frames → observations → trip fixes →
trip close, `GmApi.kt`, retries with backoff; every call is idempotent).

**Status: skeleton, not compiled on the dev laptop** (no Android SDK there). The JSON it builds follows
`edge/schema/*.schema.json` field by field; the Python simulator exercises the same protocol against the API.

## Build
Needs JDK 17 and the Android SDK (compileSdk 34). The Gradle wrapper jar is not checked in (binary):
```bash
cd edge/android
gradle wrapper --gradle-version 8.7          # once, with a system Gradle; or open the folder in Android Studio
./gradlew assembleDebug -PgmApi=http://192.168.1.10:8000 -PgmCalib=bus_default
adb install app/build/outputs/apk/debug/app-debug.apk
```
`-PgmApi` defaults to `http://10.0.2.2:8000` (the host machine from the Android emulator). The demo API must run
with `OPEN_DEVICE_REGISTRATION` (DEMO_MODE default) for the app to register itself.

## What is a stub
| Part | Now | To do |
|---|---|---|
| Segmenter | `StubSegmenter` (finds nothing) | LiteRT YOLO26n-seg from `ml/export.py --formats litert16`, GPU delegate, post-processing as in `ml/segmenter.py:ONNXSegmenter`, IoU tracker for `track_id`, ≤ 3 keyframes per track |
| Outbox | JSON-lines files in `filesDir/outbox` | Room tables with the same API |
| Config | constants (4 m, 5 km/h) | `GET /v1/edge/config` with ETag, verify the ed25519 signature, apply thresholds and `upload_wifi_only` |
| IMU | none | accelerometer `az` RMS/peak → `source: jolt` observations + ±1 s window `.npz` |
| Privacy | none | face/plate blur before JPEG encode (YuNet / plate detector) |
| Runtime | activity | foreground service (camera + location), boot start, thermal throttling |

Calibration of the mounted phone (height, pitch, intrinsics) follows `docs/CALIBRATION_SOP.md`; the `calib_id`
must exist on the server (`configs/cameras/*.json` or the `calib_cert` table).
