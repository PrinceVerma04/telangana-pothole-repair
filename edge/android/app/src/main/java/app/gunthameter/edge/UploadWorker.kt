package app.gunthameter.edge

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Drains the [Outbox] when the network allows (WorkManager retries with backoff after a failure). */
class UploadWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val prefs = applicationContext.getSharedPreferences("gm", Context.MODE_PRIVATE)
        val deviceId = prefs.getString("device_id", null) ?: return@withContext Result.failure()
        val api = GmApi(BuildConfig.GM_API)
        try {
            val token = prefs.getString("token", null) ?: api.register(deviceId, BuildConfig.CALIB_ID, null).also {
                prefs.edit().putString("token", it).apply()
            }
            Outbox(applicationContext.filesDir).drain(api, token, deviceId)
            Result.success()
        } catch (e: ApiException) {
            if (e.status == 401 || e.status == 403) prefs.edit().remove("token").apply()
            Result.retry()
        } catch (e: Exception) {
            Result.retry()
        }
    }

    companion object {
        fun enqueue(ctx: Context, wifiOnly: Boolean) {
            val req = OneTimeWorkRequestBuilder<UploadWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
                        .build()
                )
                .build()
            WorkManager.getInstance(ctx).enqueueUniqueWork("gm-upload", ExistingWorkPolicy.APPEND_OR_REPLACE, req)
        }
    }
}
