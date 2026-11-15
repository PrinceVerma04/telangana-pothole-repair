package app.gunthameter.edge

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Bus dashcam screen (skeleton): camera preview, 1 Hz GNSS trip fixes, distance-triggered keyframes run through
 * the [Segmenter], observations into the [Outbox], "End trip" closes the trip and schedules the [UploadWorker].
 * Not done here (see edge/android/README.md): signed config fetch, IMU jolts, face/plate blur, foreground service.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var outbox: Outbox
    private lateinit var location: FusedLocationProviderClient
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val segmenter: Segmenter = StubSegmenter()
    private val trigger = KeyframeTrigger()

    @Volatile private var lastFix: GpsFix? = null
    @Volatile private var tripId: String = Payloads.newUuid()
    @Volatile private var seq = 0
    @Volatile private var cameraOn = false
    private lateinit var deviceId: String

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc = result.lastLocation ?: return
            val fix = GpsFix(loc.time, loc.latitude, loc.longitude, loc.accuracy, loc.speed, loc.bearing)
            lastFix = fix
            outbox.putFix(tripId, Payloads.tripFix(fix, azRms = 0.0, camActive = cameraOn))
            runOnUiThread { status.text = "trip ${tripId.take(8)}  kf $seq  ±${loc.accuracy.toInt()} m" }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = getSharedPreferences("gm", Context.MODE_PRIVATE)
        deviceId = prefs.getString("device_id", null) ?: ("phone-" + Payloads.newUuid().take(8)).also {
            prefs.edit().putString("device_id", it).apply()
        }
        outbox = Outbox(filesDir)
        location = LocationServices.getFusedLocationProviderClient(this)

        preview = PreviewView(this)
        status = TextView(this).apply { setPadding(24, 24, 24, 24); setTextColor(0xFFFFFFFF.toInt()) }
        val endTrip = Button(this).apply {
            text = "End trip / యాత్ర ముగించు"
            setOnClickListener { endTrip() }
        }
        setContentView(FrameLayout(this).apply {
            addView(preview)
            addView(status, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.START))
            addView(endTrip, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.END))
        })

        val needed = arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION)
        if (needed.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            start()
        } else {
            ActivityCompat.requestPermissions(this, needed, 1)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            start()
        } else {
            status.text = "camera + location permission needed"
        }
    }

    @SuppressLint("MissingPermission")
    private fun start() {
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
        location.requestLocationUpdates(req, locationCallback, Looper.getMainLooper())

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val pv = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image -> analyze(image) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, pv, analysis)
            cameraOn = true
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        try {
            val fix = lastFix ?: return
            if (!trigger.shouldFire(fix.lat, fix.lon, fix.speedMps.toDouble())) return
            val bmp: Bitmap = image.toBitmap()
            val dets = segmenter.predict(bmp)
            if (dets.isEmpty()) return
            seq += 1
            val key = Payloads.frameKey(deviceId, tripId, seq)
            val jpeg = ByteArrayOutputStream().use { bos ->
                bmp.compress(Bitmap.CompressFormat.JPEG, JPEG_Q, bos)
                bos.toByteArray()
            }
            outbox.putFrame(key, jpeg)
            outbox.putObservation(
                Payloads.observation(
                    deviceId = deviceId, tripId = tripId, seq = seq, capturedAtMs = System.currentTimeMillis(),
                    monoMs = SystemClock.elapsedRealtime(), calibId = BuildConfig.CALIB_ID,
                    modelVersion = segmenter.modelVersion, fix = fix, detections = dets, frameKey = key,
                    w = bmp.width, h = bmp.height, jpegQ = JPEG_Q,
                )
            )
        } finally {
            image.close()
        }
    }

    private fun endTrip() {
        outbox.closeTrip(deviceId, tripId, Payloads.isoIst(System.currentTimeMillis()))
        UploadWorker.enqueue(this, wifiOnly = false)
        tripId = Payloads.newUuid()
        seq = 0
    }

    override fun onDestroy() {
        location.removeLocationUpdates(locationCallback)
        analysisExecutor.shutdown()
        segmenter.close()
        super.onDestroy()
    }

    companion object {
        const val JPEG_Q = 75
    }
}
