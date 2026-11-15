package app.gunthameter.edge

import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

/** One detection in raw frame pixels (edge/schema/observation.schema.json detections[]). */
data class Detection(
    val trackId: Int?,
    val cls: String,
    val conf: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val polygon: List<Pair<Float, Float>>? = null,
)

/** A GNSS fix as the location provider gives it. */
data class GpsFix(
    val epochMs: Long,
    val lat: Double,
    val lon: Double,
    val accuracyM: Float,
    val speedMps: Float,
    val headingDeg: Float,
)

/** JSON builders for the frozen wire format (edge/schema/*.schema.json). Times are sent in IST (+05:30). */
object Payloads {
    private val IST: ZoneOffset = ZoneOffset.ofHoursMinutes(5, 30)
    private val FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX")

    fun isoIst(epochMs: Long): String = OffsetDateTime.ofInstant(Instant.ofEpochMilli(epochMs), IST).format(FMT)

    fun newUuid(): String = UUID.randomUUID().toString()

    fun frameKey(deviceId: String, tripId: String, seq: Int): String = "frames/$deviceId/$tripId/$seq.jpg"

    private fun gps(fix: GpsFix, nowMs: Long): JSONObject = JSONObject()
        .put("lat", fix.lat)
        .put("lon", fix.lon)
        .put("accuracy_m", fix.accuracyM.toDouble().coerceIn(0.0, 500.0))
        .put("speed_mps", fix.speedMps.toDouble().coerceIn(0.0, 60.0))
        .put("heading_deg", ((fix.headingDeg % 360f) + 360f) % 360f)
        .put("fix_age_ms", (nowMs - fix.epochMs).coerceAtLeast(0L))

    private fun detection(d: Detection): JSONObject {
        val o = JSONObject()
            .put("track_id", d.trackId ?: JSONObject.NULL)
            .put("cls", d.cls)
            .put("conf", d.conf.toDouble().coerceIn(0.0, 1.0))
            .put("bbox", JSONArray(listOf(d.x1, d.y1, d.x2, d.y2).map { it.toDouble() }))
        val poly = d.polygon
        if (poly != null && poly.size >= 3) {
            o.put("polygon", JSONArray(poly.map { JSONArray(listOf(it.first.toDouble(), it.second.toDouble())) }))
        } else {
            o.put("polygon", JSONObject.NULL)
        }
        return o
    }

    fun observation(
        deviceId: String, tripId: String, seq: Int, capturedAtMs: Long, monoMs: Long, calibId: String,
        modelVersion: String, fix: GpsFix, detections: List<Detection>, frameKey: String?, w: Int, h: Int,
        jpegQ: Int, azRms: Double? = null, azPeakG: Double? = null,
    ): JSONObject {
        val o = JSONObject()
            .put("client_uuid", newUuid())
            .put("device_id", deviceId)
            .put("trip_id", tripId)
            .put("seq", seq)
            .put("captured_at", isoIst(capturedAtMs))
            .put("mono_ms", monoMs)
            .put("source", if (frameKey == null) "jolt" else "edge")
            .put("calib_id", calibId)
            .put("model_version", modelVersion)
            .put("gps", gps(fix, capturedAtMs))
            .put("detections", JSONArray(detections.take(32).map { detection(it) }))
        if (frameKey != null) {
            o.put("frame", JSONObject().put("key", frameKey).put("w", w).put("h", h).put("jpeg_q", jpegQ))
        } else {
            o.put("frame", JSONObject.NULL)
        }
        if (azRms != null) {
            o.put("imu", JSONObject().put("az_rms_1s", azRms).put("az_peak_g", azPeakG ?: 0.0)
                .put("window_key", JSONObject.NULL))
        } else {
            o.put("imu", JSONObject.NULL)
        }
        return o
    }

    fun tripFix(fix: GpsFix, azRms: Double, camActive: Boolean): JSONObject = JSONObject()
        .put("ts", isoIst(fix.epochMs))
        .put("lat", fix.lat)
        .put("lon", fix.lon)
        .put("accuracy_m", fix.accuracyM.toDouble().coerceIn(0.0, 500.0))
        .put("speed_mps", fix.speedMps.toDouble().coerceIn(0.0, 60.0))
        .put("heading_deg", ((fix.headingDeg % 360f) + 360f) % 360f)
        .put("az_rms", azRms.coerceAtLeast(0.0))
        .put("cam_active", camActive)
}
