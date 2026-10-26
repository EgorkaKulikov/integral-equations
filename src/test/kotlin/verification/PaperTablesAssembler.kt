package verification

import java.io.File
import kotlin.test.Test

/**
 * ASSEMBLY OF THE ARTICLE TABLES FROM THE RAW RESULTS IN build/sec4/<tag>.json.
 *
 * Run by the task `paperTables` after `sec4Tables`; writes one TSV per numerical object of the
 * article into build/paper-tables/ in the layout of the article: a block per scheme family and
 * space, rows by n, columns E_h and p_h per refinement scheme (Nystrom tables: and the ratio G).
 * The printing rule of the article is applied: an error below 1e-12 is printed as `---`, and so is
 * an observed order computed from such an error; the last row of a block has no order.
 * The raw values with all digits stay in build/sec4/<tag>.txt and .json.
 */
class PaperTablesAssembler {

    private val dir = File(System.getProperty("user.dir")).resolve("build/sec4")
    private val out = File(System.getProperty("user.dir")).resolve("build/paper-tables")
    private val r = HashMap<String, Double?>()
    private val THR = 1e-12

    private fun load() {
        val line = Regex("""^\s*"(.+)":\s*(\S+?),?\s*$""")
        dir.listFiles { f -> f.name.endsWith(".json") }!!.forEach { f ->
            f.readLines().forEach { l ->
                line.find(l)?.let { m -> r[m.groupValues[1]] = m.groupValues[2].toDoubleOrNull() }
            }
        }
    }

    private fun e(v: Double?): String = when {
        v == null -> "NaN"
        v < THR -> "---"
        else -> String.format("%.3e", v)
    }

    private fun ph(eh: Double?, ehNext: Double?, p: Double?): String =
        if (eh == null || ehNext == null || p == null || p.isNaN() || eh < THR || ehNext < THR) "---"
        else String.format("%.2f", p)

    private class Block(
        val title: String, val tab: String, val sys: String, val fam: String,
        val ns: List<Int>, val schemes: List<String>, val gRef: String? = null,
    )

    private fun table(name: String, blocks: List<Block>) {
        val sb = StringBuilder()
        for (b in blocks) {
            sb.append("# ").append(b.title).append('\n')
            sb.append("n")
            for (s in b.schemes) {
                sb.append("\t$s:E_h\t$s:p_h")
                if (b.gRef != null) sb.append("\t$s:G")
            }
            sb.append('\n')
            for ((k, n) in b.ns.withIndex()) {
                sb.append(n)
                for (s in b.schemes) {
                    val key = { nn: Int, q: String -> "${b.tab}|${b.sys}|${b.fam}|$nn|$s|$q" }
                    val eh = r[key(n, "eh")]
                    val next = if (k + 1 < b.ns.size) r[key(b.ns[k + 1], "eh")] else null
                    sb.append('\t').append(e(eh)).append('\t').append(ph(eh, next, r[key(n, "ph")]))
                    if (b.gRef != null) {
                        val ref = r["${b.tab}|${b.gRef}|$n|$s|eh"]
                        val g = if (ref == null || eh == null || ref < THR || eh < THR) "---"
                        else String.format("%.3f", ref / eh)
                        sb.append('\t').append(g)
                    }
                }
                sb.append('\n')
            }
            sb.append('\n')
        }
        File(out, "$name.tsv").writeText(sb.toString())
        println("written: " + File(out, "$name.tsv").absolutePath)
    }

    @Test
    fun assemble() {
        load()
        require(r.isNotEmpty()) { "no results in $dir: run `./gradlew sec4Tables` first" }
        out.mkdirs()
        val four = listOf("base", "sloan", "kulkarni", "iterKulkarni")
        val ny = listOf("combinedNystrom", "iterCombinedNystrom")
        val n8 = listOf(8, 16, 32, 64)
        val n32 = listOf(32, 64, 128)
        table("tab-h1h2", listOf(
            Block("M1, theta(B)", "tab:h1h2", "B", "theta", n8, four),
            Block("M1, xi1(B)", "tab:h1h2", "B", "xi1", n8, four),
            Block("M1, xi1(H1)", "tab:h1h2", "H", "xi1", n8, four),
        ))
        table("tab-h3-m2", listOf(
            Block("M2, theta(B)", "tab:h3-m2", "B", "theta", n32, four),
            Block("M2, xi1(T40)", "tab:h3-m2", "T40.0", "xi1", n32, four),
            Block("M2, theta(T40)", "tab:h3-m2", "T40.0", "theta", n32, four),
            Block("M3, theta(B)", "tab:h3-m3", "B", "theta", n32, four),
            Block("M3, xi1(T30)", "tab:h3-m3", "T30.0", "xi1", n32, four),
            Block("M3, theta(T30)", "tab:h3-m3", "T30.0", "theta", n32, four),
        ))
        table("tab-nonuni", listOf(
            Block("M2, theta(B), grid (6.1)", "tab:nonuni-m2", "B", "theta", n32, four),
            Block("M2, xi1(T40), grid (6.1)", "tab:nonuni-m2", "T40.0", "xi1", n32, four),
            Block("M2, theta(T40), grid (6.1)", "tab:nonuni-m2", "T40.0", "theta", n32, four),
            Block("M3, theta(B), grid (6.1)", "tab:nonuni-m3", "B", "theta", n32, four),
            Block("M3, xi1(T30), grid (6.1)", "tab:nonuni-m3", "T30.0", "xi1", n32, four),
            Block("M3, theta(T30), grid (6.1)", "tab:nonuni-m3", "T30.0", "theta", n32, four),
        ))
        table("tab-first-kind", listOf(
            Block("M4, xi1(B), alpha=1e-10", "tab:m4", "B", "xi1", listOf(16, 32, 64, 128), listOf("base", "sloan")),
            Block("M4, xi1(T20), alpha=1e-10", "tab:m4", "T20.0", "xi1", listOf(16, 32, 64, 128), listOf("base", "sloan")),
            Block("M5, xi1(B)", "tab:m5", "B", "xi1", n32, four),
            Block("M5, xi1(T30)", "tab:m5", "T30.0", "xi1", n32, four),
        ))
        table("tab-nystrom", listOf(
            Block("M1, theta(B)", "tab:h4c-m1", "B", "theta", listOf(4, 8, 16), ny),
            Block("M1, xi~1(B)", "tab:h4c-m1", "B", "xit1", listOf(4, 8, 16, 32), ny),
            Block("M1, xi~1(H1)", "tab:h4c-m1", "H", "xit1", listOf(4, 8, 16, 32), ny),
            Block("M3, theta(B)", "tab:h4c-m3", "B", "theta", listOf(16, 32, 64, 128), ny),
            Block("M3, theta(T30); G = theta(B)/theta(T30)", "tab:nystrom-theta-c", "T30.0", "theta",
                listOf(16, 32, 64), ny, gRef = "B|theta"),
            Block("M3, control theta(H1); G = theta(B)/theta(H1)", "tab:nystrom-theta-c", "H", "theta",
                listOf(16, 32, 64), ny, gRef = "B|theta"),
            Block("M3, xi~1(B)", "tab:h4c-m3", "B", "xit1", listOf(16, 32, 64, 128), ny),
            Block("M3, xi~1(T30)", "tab:h4c-m3", "T30.0", "xit1", listOf(16, 32, 64, 128), ny),
        ))
        solFreq()
        freq()
        m4Split()
    }

    private fun solFreq() {
        val sb = StringBuilder("# G = E_h[theta(B)]/E_h[theta(T_w0)], n = 64\nproblem\tsolution\tbase\tsloan\tkulkarni\titerKulkarni\n")
        for (p in listOf("M2", "M3")) for (u in (1..7).map { "u$it" }) {
            val cells = listOf("base", "sloan", "kulkarni", "iterKulkarni").map { r["tab:sol-freq|$p|$u|64|$it|G"] }
            if (cells.all { it == null }) continue
            sb.append("$p\t$u\t").append(cells.joinToString("\t") { if (it == null) "NaN" else String.format("%.4g", it) }).append('\n')
        }
        File(out, "tab-sol-freq.tsv").writeText(sb.toString())
    }

    private fun freq() {
        val sb = StringBuilder("# fig:freq: G for T_w, w/w0 on the abscissa, n = 64\nw/w0\tM2 basic\tM2 Kulkarni\tM3 basic\tM3 Kulkarni\n")
        for (x in listOf("0.25", "0.50", "0.75", "0.90", "1.00", "1.05", "1.10", "1.25", "1.40", "1.50", "2.00")) {
            val c = listOf("M2|$x|base", "M2|$x|kulkarni", "M3|$x|base", "M3|$x|kulkarni").map { r["tab:freq|$it"] }
            sb.append(x).append('\t').append(c.joinToString("\t") { if (it == null) "NaN" else String.format("%.3g", it) }).append('\n')
        }
        File(out, "fig-freq.tsv").writeText(sb.toString())
    }

    private fun m4Split() {
        val sb = StringBuilder("# rem:m4-split: ||u* - u_alpha||_inf (sup over 12801 points; max over nodes)\nalpha\tNsub\tsup\tnodes\n")
        for (a in listOf("1e-06", "1e-08", "1e-10", "1e-12")) for (ns in listOf(64, 128, 256)) {
            val s = r["rem:m4-split|$a|$ns|sup"] ?: continue
            sb.append("$a\t$ns\t").append(String.format("%.4e", s)).append('\t')
                .append(String.format("%.4e", r["rem:m4-split|$a|$ns|nodes"] ?: Double.NaN)).append('\n')
        }
        File(out, "rem-m4-split.tsv").writeText(sb.toString())
    }
}
