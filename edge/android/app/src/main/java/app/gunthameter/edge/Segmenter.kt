package app.gunthameter.edge

import android.graphics.Bitmap

/**
 * On-device pothole segmenter. Production: YOLO26n-seg exported by `ml/export.py --formats litert16` and run
 * with LiteRT (GPU delegate), letterbox 640, class-aware NMS, prototype masks -> polygons; same post-processing
 * as ml/segmenter.py:ONNXSegmenter. The skeleton ships [StubSegmenter] so the capture/upload path can be tested
 * without a model.
 */
interface Segmenter {
    val modelVersion: String
    fun predict(frame: Bitmap): List<Detection>
    fun close() {}
}

/** Finds nothing (keeps the app runnable before a model is bundled in assets/). */
class StubSegmenter : Segmenter {
    override val modelVersion: String = "stub@android-0.1"
    override fun predict(frame: Bitmap): List<Detection> = emptyList()
}
