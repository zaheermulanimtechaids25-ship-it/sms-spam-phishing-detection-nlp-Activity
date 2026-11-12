// SmsPhishingClassifier.kt
// On-device SMS classifier using ONNX Runtime Mobile + ONNX Runtime Extensions.
//
// STATUS: reference integration code. The ONNX model it loads (sms_classifier_e2e.onnx) was
// built and verified with ONNX Runtime + onnxruntime-extensions on desktop (see
// results/metrics/onnx_export_report.json). This Kotlin file has NOT been compiled or run on
// a physical Android device by the project team.
//
// Gradle (app/build.gradle.kts):
//   implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
//   implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.13.0")
// Assets: copy models/optimized/sms_classifier_e2e.onnx into app/src/main/assets/
//
// Usage:
//   val clf = SmsPhishingClassifier(context)
//   val result = clf.classify("Aapka SBI account band ho jayega. KYC update karo")
//   // result.label == "phishing", result.scores["phishing"] == 0.98 ...

package com.example.smsguard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.content.Context
import java.text.Normalizer
import kotlin.math.exp

data class SmsResult(val label: String, val confidence: Float, val scores: Map<String, Float>)

class SmsPhishingClassifier(context: Context, modelAsset: String = "sms_classifier_e2e.onnx") : AutoCloseable {
    private val labels = listOf("ham", "spam", "phishing")
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions().apply {
            registerCustomOpLibrary(OrtxPackage.getLibraryPath())   // SentencePiece tokenizer op
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        session = env.createSession(bytes, opts)
    }

    fun classify(sms: String): SmsResult {
        val text = Preprocess.apply(sms)
        OnnxTensor.createTensor(env, arrayOf(text), longArrayOf(1)).use { input ->
            session.run(mapOf("text" to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val logits = (out[0].value as Array<FloatArray>)[0]
                val max = logits.maxOrNull() ?: 0f
                val e = logits.map { exp((it - max).toDouble()) }
                val sum = e.sum()
                val probs = e.map { (it / sum).toFloat() }
                val i = probs.indices.maxByOrNull { probs[it] } ?: 0
                return SmsResult(labels[i], probs[i], labels.zip(probs).toMap())
            }
        }
    }

    override fun close() {
        session.close()
    }
}

/** Kotlin port of deployment/mobile_inference.py::preprocess (must stay identical to training). */
object Preprocess {
    private val mojibake = linkedMapOf(
        "ï¿½" to "", "�" to "", "Â£" to "£", "Â" to "", "â‚¬" to "€", "â€™" to "'", "â€˜" to "'",
        "â€œ" to "\"", "â€\u009d" to "\"", "â€“" to "-", "â€”" to "-", "â€¦" to "...",
        "&lt;" to "<", "&gt;" to ">", "&amp;" to "&", "&#039;" to "'"
    )
    private val email = Regex("""\b[\w.+-]+@[\w-]+\.[\w.-]+\b""")
    private val url = Regex(
        """(?:(?:https?://|www\.)\S*)|(?:\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|in|co|uk|io|ly|me|info|biz|""" +
        """mobi|xyz|top|online|site|click|link|example|test|do|gl|cc|tk|ml|ga|cf|gq|app|live|club|win)(?:/\S*)?)""",
        RegexOption.IGNORE_CASE
    )
    private val phone = Regex("""(?<!\w)(?:\+?\d[\d\s-]{7,}\d)(?!\w)""")
    private val glued = Regex("""\d{7,}""")
    private val num = Regex("""(?<![\w<])\d{4,}(?![\w>])""")
    private val repeat = Regex("""(.)\1{2,}""")
    private val punct = Regex("""([!?.])\1+""")
    private val space = Regex("""\s+""")

    fun apply(raw: String): String {
        var t = raw
        for ((bad, good) in mojibake) t = t.replace(bad, good)
        t = Normalizer.normalize(t, Normalizer.Form.NFKC).trim()
        t = email.replace(t, " <email> ")
        t = url.replace(t, " <url> ")
        t = phone.replace(t, " <phone> ")
        t = glued.replace(t, " <phone> ")
        t = num.replace(t, " <num> ")
        t = space.replace(t, " ").trim()
        t = repeat.replace(t, "$1$1")
        t = punct.replace(t, "$1")
        return space.replace(t, " ").trim()
    }
}
