package app.gunthameter.edge

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ApiException(val status: Int, body: String) : IOException("HTTP $status: ${body.take(300)}")

/**
 * Minimal GunthaMeter API client (edge/protocol.md). Every call here is idempotent, so transport errors, 429 and
 * 5xx are retried with exponential backoff, re-sending the same payload. Mirrors edge/sim/client.py.
 */
class GmApi(baseUrl: String, private val retries: Int = 4) {
    private val base = baseUrl.trimEnd('/')
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()
    private val json = "application/json".toMediaType()

    private fun call(build: () -> Request): String {
        var last: IOException? = null
        for (attempt in 0..retries) {
            if (attempt > 0) Thread.sleep(500L shl (attempt - 1))
            try {
                http.newCall(build()).execute().use { r: Response ->
                    val body = r.body?.string() ?: ""
                    if (r.isSuccessful) return body
                    val e = ApiException(r.code, body)
                    if (r.code != 429 && r.code < 500) throw e
                    last = e
                }
            } catch (e: ApiException) {
                if (e.status != 429 && e.status < 500) throw e
                last = e
            } catch (e: IOException) {
                last = e
            }
        }
        throw last ?: IOException("request failed")
    }

    private fun post(path: String, body: JSONObject, token: String?): JSONObject {
        val text = call {
            Request.Builder().url("$base$path").post(body.toString().toRequestBody(json)).apply {
                if (token != null) header("Authorization", "Bearer $token")
            }.build()
        }
        return if (text.isBlank()) JSONObject() else JSONObject(text)
    }

    fun register(deviceId: String, calibId: String, vehicleReg: String?, agencyCode: String = "GHMC"): String {
        val body = JSONObject().put("device_id", deviceId).put("kind", "phone").put("agency_code", agencyCode)
            .put("calib_id", calibId).put("vehicle_reg", vehicleReg ?: JSONObject.NULL)
        return post("/v1/edge/register", body, null).getString("token")
    }

    fun presign(token: String, keys: List<String>): Map<String, String> {
        val res = post("/v1/edge/presign", JSONObject().put("keys", JSONArray(keys)).put("content_type", "image/jpeg"),
            token)
        val urls = res.getJSONObject("urls")
        return keys.associateWith { urls.getString(it) }
    }

    fun put(url: String, data: ByteArray) {
        val full = if (url.startsWith("http")) url else "$base$url"
        call { Request.Builder().url(full).put(data.toRequestBody("image/jpeg".toMediaType())).build() }
    }

    fun postObservations(token: String, observations: List<JSONObject>): JSONObject =
        post("/v1/ingest/observations", JSONObject().put("observations", JSONArray(observations)), token)

    fun postTripFixes(token: String, deviceId: String, tripId: String, fixes: List<JSONObject>): JSONObject =
        post("/v1/ingest/trip-fixes",
            JSONObject().put("device_id", deviceId).put("trip_id", tripId).put("fixes", JSONArray(fixes)), token)

    fun tripClose(token: String, deviceId: String, tripId: String, endedAtIso: String): JSONObject =
        post("/v1/ingest/trip-close",
            JSONObject().put("device_id", deviceId).put("trip_id", tripId).put("ended_at", endedAtIso), token)
}
