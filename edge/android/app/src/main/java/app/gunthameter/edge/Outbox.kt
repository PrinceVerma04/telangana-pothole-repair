package app.gunthameter.edge

import org.json.JSONObject
import java.io.File

/**
 * Durable store-and-forward queue (edge/protocol.md section 2.7). Skeleton: append-only JSON-lines files plus frame
 * files under filesDir/outbox; production swaps this for Room with the same API. An item is deleted only after the
 * server answered, and every item keeps its client_uuid, so a resend after a crash is deduplicated by the server.
 *
 * Layout: frames/<key with '/' replaced by '__'>, observations.jsonl, fixes/<trip>.jsonl, closed/<trip>.json
 */
class Outbox(root: File) {
    private val dir = File(root, "outbox").apply { mkdirs() }
    private val framesDir = File(dir, "frames").apply { mkdirs() }
    private val fixesDir = File(dir, "fixes").apply { mkdirs() }
    private val closedDir = File(dir, "closed").apply { mkdirs() }
    private val observations = File(dir, "observations.jsonl")

    @Synchronized
    fun putFrame(key: String, jpeg: ByteArray) {
        File(framesDir, key.replace("/", "__")).writeBytes(jpeg)
    }

    @Synchronized
    fun putObservation(o: JSONObject) {
        observations.appendText(o.toString() + "\n")
    }

    @Synchronized
    fun putFix(tripId: String, fix: JSONObject) {
        File(fixesDir, "$tripId.jsonl").appendText(fix.toString() + "\n")
    }

    @Synchronized
    fun closeTrip(deviceId: String, tripId: String, endedAtIso: String) {
        File(closedDir, "$tripId.json").writeText(
            JSONObject().put("device_id", deviceId).put("trip_id", tripId).put("ended_at", endedAtIso).toString()
        )
    }

    /** Upload everything in protocol order: frames -> observations -> fixes -> trip close. Throws on failure; what
     * was acknowledged before the failure is already removed, the rest is retried by the next run. */
    @Synchronized
    fun drain(api: GmApi, token: String, deviceId: String, batch: Int = 50) {
        // 1. frames (presign in chunks of 50, then PUT)
        val frames = framesDir.listFiles()?.sortedBy { it.name } ?: emptyList()
        frames.chunked(50).forEach { chunk ->
            val keys = chunk.map { it.name.replace("__", "/") }
            val urls = api.presign(token, keys)
            chunk.zip(keys).forEach { (f, k) ->
                api.put(urls.getValue(k), f.readBytes())
                f.delete()
            }
        }
        // 2. observations (idempotent by client_uuid)
        if (observations.exists()) {
            val lines = observations.readLines().filter { it.isNotBlank() }
            lines.chunked(batch).forEachIndexed { i, chunk ->
                api.postObservations(token, chunk.map { JSONObject(it) })
                val rest = lines.drop((i + 1) * batch)
                observations.writeText(if (rest.isEmpty()) "" else rest.joinToString("\n", postfix = "\n"))
            }
        }
        // 3. fixes per trip, 4. trip close once its fixes are in
        fixesDir.listFiles()?.forEach { f ->
            val tripId = f.nameWithoutExtension
            val fixes = f.readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
            fixes.chunked(600).forEach { api.postTripFixes(token, deviceId, tripId, it) }
            f.delete()
        }
        closedDir.listFiles()?.forEach { f ->
            val c = JSONObject(f.readText())
            api.tripClose(token, c.getString("device_id"), c.getString("trip_id"), c.getString("ended_at"))
            f.delete()
        }
    }
}
