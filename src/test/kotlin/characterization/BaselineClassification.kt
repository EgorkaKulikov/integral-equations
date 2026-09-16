package characterization

import kotlin.math.abs

/**
 * THE CLASS OF A BASELINE ROW — the comparison mode taken FROM THE DATA and not from the suffix of the key.
 *
 * Before that there were five modes, and they were chosen by a chain of `if`s on the name of the key
 * ([ExtraCharacterizationTest] branched on `.residual`) or were not chosen at all
 * ([EhCharacterizationTest] compared everything with one tolerance). The class moves the decision
 * into the third column of the TSV, while the column itself is COMPUTED by the tool [BaselineClassifier]
 * from two snapshots (`-Dnumerics.backend=java` and `native`): there is no handwritten list
 * of "what fails today" in the project.
 *
 * The classes and their justification (all the numbers are measurements, see `docs/baseline-changes.md`):
 *  - [PORTABLE] — the value is portable between the LU paths: the discrepancy of the backends is not above
 *    `1.0e-14` at the floor `6e-13` (a 60x margin). The rule is the former one: rel. 1e-9 at the floor 6e-13;
 *  - [SENSITIVE] — the value is bound to the LU path: the problem F1 (Wazwaz regularization,
 *    `alpha = 1e-10`, `cond_1 ~ 2.2e10`). It is compared against the bound COMPUTED in the run,
 *    `2*cond*max(omega,eps)*||u||inf` ([F1SystemConditioning]), and not against a constant:
 *    a constant would bind the baseline to a machine again;
 *  - [RESIDUAL] — the residual of the iterative schemes: rel. 1e-3 at the noise 6e-15 (the former
 *    behaviour of the `*.residual` keys);
 *  - [EXACT] — the integral iteration counters: a strict equality of the strings. The tightening
 *    is permitted by the measurement: `.iters` never diverged (0 of 336 keys).
 */
enum class BaselineClass(val tag: String) {
    PORTABLE("portable"),
    SENSITIVE("sensitive"),
    RESIDUAL("residual"),
    EXACT("exact"),
    ;

    companion object {
        /** The class by the tag from the column; an unknown tag is an ERROR and not a "default value". */
        fun ofTag(tag: String): BaselineClass =
            entries.firstOrNull { it.tag == tag }
                ?: error(
                    "unknown class of a baseline row '$tag'; admissible: " +
                        entries.joinToString(", ") { it.tag },
                )
    }
}

/** A baseline row: the recorded value (as a string) and the comparison class. */
data class BaselineEntry(val value: String, val cls: BaselineClass)

/**
 * THE BASELINE FORMAT AND THE SINGLE PLACE OF COMPARISON for both characterization gates.
 *
 * The row format: `key <TAB> value <TAB> class`. Comment lines start with `#`,
 * empty ones are ignored; ANY other line must split into exactly three parts —
 * otherwise the parsing fails. The former parser ("exactly two parts, otherwise the line is silently treated
 * as a comment") turned a typo in the format into "the key is absent from the baseline", that is,
 * it weakened the gate silently.
 *
 * The tolerances are NOT duplicated: the numbers are taken from [ExtraCharacterizationMatrix], where they are
 * accompanied by a measured justification.
 */
object BaselineFormat {
    /** The rel. tolerance of the class [BaselineClass.PORTABLE]. */
    const val PORTABLE_RELATIVE_TOLERANCE = ExtraCharacterizationMatrix.RELATIVE_TOLERANCE

    /** The absolute "floor" of the class [BaselineClass.PORTABLE]: `10^3*eps*||u||inf` at `||u||inf ~ e`. */
    const val PORTABLE_ABSOLUTE_FLOOR = ExtraCharacterizationMatrix.NOISE_FLOOR

    /** The rel. tolerance of the class [BaselineClass.RESIDUAL]. */
    const val RESIDUAL_RELATIVE_TOLERANCE = ExtraCharacterizationMatrix.RESIDUAL_RELATIVE_TOLERANCE

    /** The noise of the class [BaselineClass.RESIDUAL]: below it a discrepancy is not checked. */
    const val RESIDUAL_NOISE_FLOOR = ExtraCharacterizationMatrix.RESIDUAL_NOISE_FLOOR

    /** The "floor" of the relative measure of the class [BaselineClass.RESIDUAL]. */
    const val RESIDUAL_ABSOLUTE_FLOOR = ExtraCharacterizationMatrix.RESIDUAL_ABSOLUTE_FLOOR

    /** The suffix of the keys with an iteration counter (the class [BaselineClass.EXACT]). */
    const val ITERATIONS_SUFFIX = ".iters"

    /** Parses the baseline file. A format violation or an unknown class is a failure. */
    fun parse(lines: Sequence<String>, source: String): Map<String, BaselineEntry> {
        val result = LinkedHashMap<String, BaselineEntry>()
        var number = 0
        for (raw in lines) {
            number++
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith("#")) continue
            val parts = line.split('\t')
            check(parts.size == 3) {
                "$source:$number: exactly three columns \"key<TAB>value<TAB>class\" were expected, " +
                    "got ${parts.size}: '$line'"
            }
            val cls = try {
                BaselineClass.ofTag(parts[2])
            } catch (e: IllegalStateException) {
                error("$source:$number: ${e.message}")
            }
            check(result.put(parts[0], BaselineEntry(parts[1], cls)) == null) {
                "$source:$number: the key '${parts[0]}' occurs in the baseline twice"
            }
        }
        check(result.isNotEmpty()) { "$source: not a single data row was parsed" }
        return result
    }

    /**
     * Compares the computed value with the baseline one BY THE RULE OF THE CLASS.
     *
     * Returns a description of the discrepancy or `null` if the values agree.
     * For the class [BaselineClass.SENSITIVE] the bound is taken from [boundFor] — it is
     * computed in the SAME run and on the SAME backend, so it does not bind
     * the baseline to a machine.
     */
    fun compare(
        key: String,
        expected: BaselineEntry,
        actual: String,
        boundFor: (String) -> F1SystemConditioning.Measurement? = { F1SystemConditioning.measureFor(it) },
    ): String? {
        val expectedValue = expected.value
        if (expected.cls == BaselineClass.EXACT) {
            return if (expectedValue == actual) {
                null
            } else {
                "$key [exact]: baseline=$expectedValue, got=$actual (a match of the strings is required)"
            }
        }
        val e = expectedValue.toDoubleOrNull()
        val a = actual.toDoubleOrNull()
        if (e == null || a == null) {
            // At least one side is a marker NaN/Infinity/ERROR:<class>: the comparison is strict, by string.
            return if (expectedValue == actual) {
                null
            } else {
                "$key: baseline=$expectedValue, got=$actual (a special value is compared strictly)"
            }
        }
        if (e.isNaN() || a.isNaN()) {
            return if (e.isNaN() == a.isNaN()) null else "$key: baseline=$expectedValue, got=$actual (NaN against a number)"
        }
        val difference = abs(a - e)
        return when (expected.cls) {
            BaselineClass.PORTABLE -> {
                if (difference <= PORTABLE_ABSOLUTE_FLOOR) {
                    null
                } else {
                    val relative = difference / maxOf(abs(e), PORTABLE_ABSOLUTE_FLOOR)
                    if (relative <= PORTABLE_RELATIVE_TOLERANCE) {
                        null
                    } else {
                        "$key [portable]: baseline=$expectedValue, got=$actual, " +
                            "rel.discrepancy=$relative (tolerance $PORTABLE_RELATIVE_TOLERANCE, floor $PORTABLE_ABSOLUTE_FLOOR)"
                    }
                }
            }
            BaselineClass.RESIDUAL -> {
                if (difference <= RESIDUAL_NOISE_FLOOR) {
                    null
                } else {
                    val relative = difference / maxOf(abs(e), RESIDUAL_ABSOLUTE_FLOOR)
                    if (relative <= RESIDUAL_RELATIVE_TOLERANCE) {
                        null
                    } else {
                        "$key [residual]: baseline=$expectedValue, got=$actual, " +
                            "rel.discrepancy=$relative (tolerance $RESIDUAL_RELATIVE_TOLERANCE, noise $RESIDUAL_NOISE_FLOOR)"
                    }
                }
            }
            BaselineClass.SENSITIVE -> {
                val m = boundFor(key)
                    ?: return "$key [sensitive]: the cond*omega bound is not computable for this key — " +
                        "the class was assigned to a key that has no available system (I-M)c=g"
                if (!m.reliable) {
                    return "$key [sensitive]: the conditioning estimate is untrustworthy " +
                        "(cond_1=${m.cond}, omega=${m.omega}, bound=${m.bound}, ||u||inf=${m.uNorm})"
                }
                if (difference <= m.bound) {
                    null
                } else {
                    "$key [sensitive]: baseline=$expectedValue, got=$actual, |dlt|=$difference > " +
                        "bound=${m.bound} (cond_1=${m.cond}, omega=${m.omega}, ||u||inf=${m.uNorm}, " +
                        "|dlt|/bound=${difference / m.bound})"
                }
            }
            BaselineClass.EXACT -> null
        }
    }
}
