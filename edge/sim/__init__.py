"""RoadFix edge simulator (WP7).

Virtual buses drive real TGSRTC GTFS route shapes (``data/interim/geo/gtfs_routes.json``) and speak the edge
protocol (``edge/protocol.md``, ``edge/schema``) to the live API: distance-triggered keyframes, detections from
dataset labels (``--use-labels``) placed geometrically on the route, GNSS noise, 1 Hz trip fixes with
``cam_active``, trip close, jolts and citizen reports.

Modules: ``geodesy`` (local ENU maths), ``route`` (GTFS shape loading, cleaning, windows), ``camera`` (pinhole +
road-plane projection of the device calibration), ``gnss`` (correlated GNSS error), ``labels`` (labelled pothole
instances and clean road backgrounds), ``scene`` (potholes placed on a route), ``frames`` (keyframe image
composition, privacy hook), ``payloads`` (observation / trip_fix dicts + JSON-Schema validation), ``client`` (HTTP
client with retries, idempotent by client_uuid), ``bus`` (one trip: simulate + upload), ``replay`` / ``fleet``
(CLIs), ``sim_server`` (local API launcher with the synthetic road-render depth measurer).
"""
