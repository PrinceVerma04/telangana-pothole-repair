package app.gunthameter.edge

import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Distance trigger (PLAN 2.2, edge/protocol.md section 2): a keyframe every [minDistanceM] metres while the bus
 * drives at >= [minSpeedKmh]. Same rule as edge/sim/bus.py and edge/jetson/runner.py. Values come from the signed
 * edge config (trigger_min_distance_m, trigger_min_speed_kmh).
 */
class KeyframeTrigger(private val minDistanceM: Double = 4.0, minSpeedKmh: Double = 5.0) {
    private val minSpeedMps = minSpeedKmh / 3.6
    private var lastLat: Double? = null
    private var lastLon: Double? = null

    @Synchronized
    fun shouldFire(lat: Double, lon: Double, speedMps: Double): Boolean {
        if (speedMps < minSpeedMps) return false
        val la = lastLat
        val lo = lastLon
        if (la == null || lo == null || haversineM(la, lo, lat, lon) >= minDistanceM) {
            lastLat = lat
            lastLon = lon
            return true
        }
        return false
    }

    companion object {
        fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val r = 6371008.8
            val p1 = Math.toRadians(lat1)
            val p2 = Math.toRadians(lat2)
            val dp = p2 - p1
            val dl = Math.toRadians(lon2 - lon1)
            val a = sin(dp / 2).pow(2) + cos(p1) * cos(p2) * sin(dl / 2).pow(2)
            return 2 * r * asin(sqrt(a))
        }
    }
}
