# Bus camera calibration — standard operating procedure (SOP)
# బస్సు కెమెరా క్యాలిబ్రేషన్ — ప్రామాణిక నిర్వహణ విధానం

Who: depot technician + one helper. Where: a flat, straight stretch of depot yard (≥ 15 m long). Time: ≈ 10 min per
bus (part B); part A is done once per camera model. Result: a calibration certificate (`calib_cert` JSON) for the
device. Without it, measurements are prioritisation-grade only (`certified=false`), never Measurement-Book (billing)
grade. Method background: `docs/DEPTH_METHOD.md` §1; code: `measure/calib.py`.

ఎవరు: డిపో టెక్నీషియన్ + ఒక సహాయకుడు. ఎక్కడ: డిపో యార్డ్‌లో సమతలంగా, నేరుగా ఉన్న కనీసం 15 మీ. స్థలం. సమయం: ఒక్కో
బస్సుకు సుమారు 10 నిమిషాలు (భాగం B); భాగం A ఒక్కో కెమెరా మోడల్‌కు ఒక్కసారే. ఫలితం: ఆ పరికరానికి క్యాలిబ్రేషన్
సర్టిఫికెట్. అది లేకపోతే కొలతలు ప్రాధాన్యత నిర్ణయానికి మాత్రమే పనికొస్తాయి, బిల్లింగ్ (మెజర్‌మెంట్ బుక్)కు కాదు.

> The Telugu text is a working translation for depot staff; have it reviewed by a native Telugu-speaking engineer
> before it is printed for field use. Where the two versions differ, the English text and the numbers govern.
> తెలుగు పాఠం డిపో సిబ్బంది కోసం చేసిన అనువాదం; ముద్రించే ముందు తెలుగు తెలిసిన ఇంజనీర్ సరిచూడాలి. తేడా ఉంటే
> ఇంగ్లీష్ పాఠం మరియు అంకెలు చెల్లుతాయి.

---

## Equipment / సామగ్రి
| Item | Spec | వస్తువు |
|---|---|---|
| Checkerboard | 9 × 6 inner corners, 35 mm squares, printed flat on A2 foam board | చెక్కర్‌బోర్డ్ (9×6, 35 మి.మీ. చతురస్రాలు, A2 ఫోమ్ బోర్డు) |
| Measuring tape | 5 m steel tape, mm marks | 5 మీ. స్టీల్ టేప్ |
| Long tape | 30 m | 30 మీ. టేప్ |
| Cones | 4 traffic cones (750 mm, IRC:67) | 4 ట్రాఫిక్ కోన్లు |
| Reference square | 1.000 m × 1.000 m, painted or taped on the ground, corners marked | నేలపై 1 మీ. × 1 మీ. చతురస్రం |
| Spirit level / plumb line | to mark the point on the ground below the camera | స్పిరిట్ లెవెల్ / ప్లంబ్ లైన్ |
| Laptop | GunthaMeter repo + `.venv` (or the tablet app, later) | ల్యాప్‌టాప్ |

## Part A — Intrinsics (once per camera model) / భాగం A — కెమెరా లోపలి విలువలు (మోడల్‌కు ఒక్కసారి)
1. Lock focus and zoom; switch off digital stabilisation and HDR. Use the resolution the bus will record at
   (1280 × 720 unless the edge config says otherwise).
   ఫోకస్, జూమ్ లాక్ చేయండి; డిజిటల్ స్టెబిలైజేషన్, HDR ఆపండి. బస్సులో వాడే రిజల్యూషన్‌నే (1280×720) వాడండి.
2. Take 20–25 photos of the checkerboard: near and far, tilted left/right/up/down, covering all corners of the frame.
   The board must be flat and sharp in every photo.
   చెక్కర్‌బోర్డ్‌ను 20–25 ఫోటోలు తీయండి: దగ్గరగా, దూరంగా, వంచి, ఫ్రేమ్ అన్ని మూలలు కవర్ అయ్యేలా. బోర్డు చదునుగా,
   ఫోటో స్పష్టంగా ఉండాలి.
3. Run `calibrate_from_images` (script in Part D). **Accept if RMS reprojection < 0.3 px.** Otherwise retake the
   blurred / badly covered photos.
   **RMS 0.3 px కన్నా తక్కువ అయితేనే అంగీకరించండి.** లేకపోతే అస్పష్టమైన ఫోటోలు మళ్లీ తీయండి.

## Part B — Mounting and extrinsics (every bus, every re-mount) / భాగం B — బస్సుపై అమరిక (ప్రతి బస్సుకు)
1. Park the bus on the flat stretch, engine off, normal tyre pressure, no passengers. Camera behind the windscreen
   in the wiper area, centred, nothing in the lower half of the view (dashboard, wipers at rest outside the view).
   బస్సును సమతల స్థలంలో ఆపండి, ఇంజన్ ఆఫ్, ప్రయాణికులు లేకుండా. కెమెరా విండ్‌స్క్రీన్ వెనుక, వైపర్ తుడిచే ప్రాంతంలో,
   మధ్యలో; చిత్రం కింది సగంలో డాష్‌బోర్డ్ కనిపించకూడదు.
2. **Camera height h_true:** plumb line from the lens centre to the ground; tape-measure the height. Record to ±1 cm
   (bus dash ≈ 2.2–2.5 m). Mark the ground point below the lens.
   **కెమెరా ఎత్తు:** లెన్స్ మధ్య నుంచి నేల వరకు ప్లంబ్ లైన్; టేప్‌తో ఎత్తు కొలవండి, ±1 సెం.మీ. వరకు రాయండి.
3. **Lateral offset:** distance from the bus centreline to the lens (+ = right). Record to ±1 cm.
   **పక్క దూరం:** బస్సు మధ్య రేఖ నుంచి లెన్స్ వరకు దూరం (కుడి = +).
4. **Cones:** place 4 cones along the lane centre at 3, 5, 8 and 12 m ahead of the ground point (long tape), and
   measure each cone's lateral offset. Take one photo from the mounted camera.
   **కోన్లు:** ముందు 3, 5, 8, 12 మీ. దూరంలో 4 కోన్లు పెట్టండి; ఒక్కోదాని పక్క దూరం కొలవండి. అమర్చిన కెమెరాతో ఒక
   ఫోటో తీయండి.
5. **Reference square:** the 1 m × 1 m square with its near edge at 5 m. Photo from the mounted camera.
   **చతురస్రం:** 1 మీ. × 1 మీ. చతురస్రం, దగ్గరి అంచు 5 మీ. వద్ద. కెమెరాతో ఫోటో తీయండి.
6. Click the cone bases and square corners in the photos (pixel coordinates) and run Part D.
   **Accept if the square reconstructs within 5 % of 1.000 m².** Otherwise re-measure the height and cone positions.
   ఫోటోలలో కోన్ అడుగులు, చతురస్రం మూలల పిక్సెల్ స్థానాలు గుర్తించి భాగం D నడపండి. **చతురస్రం వైశాల్యం 1.000 చ.మీ.కు
   5 % లోపు వస్తేనే అంగీకరించండి.**

## Part C — Records / భాగం C — రికార్డులు
Fill in for every bus (keep the sheet with the certificate):
ప్రతి బస్సుకు ఈ వివరాలు రాయండి:

| Field | Value |
|---|---|
| Depot / డిపో | |
| Bus registration / బస్సు నంబరు (e.g. TS09UA1234) | |
| Device id / పరికరం ఐడి | |
| Camera model + intrinsics file / కెమెరా మోడల్ | |
| h_true (m, ±1 cm) / ఎత్తు | |
| Lateral offset (m) / పక్క దూరం | |
| Cone offsets at 3/5/8/12 m / కోన్ దూరాలు | |
| Square error (%) / చతురస్రం దోషం | |
| Calibrated by / చేసినవారు, Date / తేదీ | |

## Part D — Compute and register the certificate / భాగం D — సర్టిఫికెట్ తయారీ
```python
# .venv\Scripts\python.exe  (from C:\gunthameter)
import cv2, numpy as np
from pathlib import Path
from measure.calib import calibrate_from_images, solve_extrinsics, known_object_error, make_cert, save_cert

intr = calibrate_from_images([cv2.imread(str(p)) for p in Path("calib/board").glob("*.jpg")])
assert intr["accepted"], intr["rms_px"]                     # RMS < 0.3 px

# road frame: X right, Y up (0 on the road), Z forward from the ground point below the lens
road = np.array([[0.05, 0, 3.0], [0.00, 0, 5.0], [-0.03, 0, 8.0], [0.02, 0, 12.0]])   # tape: cone bases
img = np.array([[641, 655], [640, 548], [639, 489], [640, 455]], float)              # clicked pixels
extr = solve_extrinsics(road, img, intr["K"], np.array(intr["dist"]))

square_px = np.array([[590, 560], [690, 560], [700, 535], [582, 535]], float)         # clicked corners
ko = known_object_error(square_px, 1.0, intr["K"], h_m=2.36, pitch_deg=extr["pitch_deg"],
                        roll_deg=extr["roll_deg"], yaw_deg=extr["yaw_deg"])

cert = make_cert("bus_ts09ua1234", "tsrtc-bus-ts09ua1234", intr, extr, vehicle_kind="bus", h_true_m=2.36,
                 known_object_err_pct=ko["err_pct"], calibrated_by="depot-tech-07", lateral_offset_m=0.10)
save_cert(cert, "configs/cameras/bus_ts09ua1234.json")
print(cert.certified, ko)
```
(The pixel and tape values above are placeholders: use the ones measured in Parts A–C.)
`make_cert` sets `certified = true` only when the intrinsics RMS < 0.3 px **and** the square is within 5 %. Then
register the device with that `calib_id` (`POST /v1/edge/register`, see `edge/protocol.md`). The certificate becomes
billing-grade only after the per-device truth fit (`measure/truth.py`, ≥ 20 inspector-measured potholes), which
fills `truth_fit` (k_sigma, IMU gain, depth MAE, volume MAPE).

## Part E — When to recalibrate / భాగం E — మళ్లీ ఎప్పుడు చేయాలి
Recalibrate (Part B; Part A only for a new camera model) when any of these happens:
ఈ సందర్భాల్లో మళ్లీ క్యాలిబ్రేషన్ చేయండి:
- the camera or its mount was moved, knocked or replaced / కెమెరా లేదా మౌంట్ కదిలినా, దెబ్బ తగిలినా, మార్చినా;
- windscreen replaced, suspension work, new tyres of a different size / విండ్‌స్క్రీన్ మార్చినా, సస్పెన్షన్ పని, వేరే
  సైజు టైర్లు;
- the server raises a **drift alert**: scale drift > 5 % (rolling camera height from the road plane vs h_true) or
  pitch drift > 3°, on several trips in a row / సర్వర్ **డ్రిఫ్ట్ అలర్ట్** ఇస్తే: ఎత్తు తేడా 5 % పైగా లేదా కోణం 3° పైగా,
  వరుసగా కొన్ని ట్రిప్పుల్లో;
- every 6 months anyway / ఏదైనా సరే ప్రతి 6 నెలలకు ఒకసారి.

Never edit a certificate by hand to make an alert go away: re-measure.
అలర్ట్ పోవడానికి సర్టిఫికెట్‌ను చేత్తో మార్చకండి — మళ్లీ కొలవండి.
