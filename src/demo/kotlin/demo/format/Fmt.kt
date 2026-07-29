package demo.format

import kotlin.math.abs

object Fmt {
    fun e(x: Double): String = if (x.isNaN()) "---" else "%.3e".format(x)
    fun p(x: Double): String = if (x.isNaN()) "---" else "%.2f".format(x)
    fun h(x: Double): String = "%.4f".format(x)
    fun tex(x: Double): String {
        if (x.isNaN()) return "---"
        if (x == 0.0) return "0"
        val exp = Math.floor(Math.log10(abs(x))).toInt()
        val mant = x / Math.pow(10.0, exp.toDouble())
        return "%.3f".format(mant).replace(".", "{,}") + "\\!\\cdot\\!10^{" + exp + "}"
    }
}
