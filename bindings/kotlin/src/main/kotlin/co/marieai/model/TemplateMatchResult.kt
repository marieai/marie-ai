package co.marieai.model

/***
 * A result from Match Template call. It specifies the page of an image as well as the area in which the was found.
 * The result also contains the label of the template and the similarity score.
 */
data class TemplateMatchResult(
    val bbox: List<Int>,
    val label: String,
    val score: Float,
    val similarity: Float,
    val frameIndex: Int? = 0
)

/***
 * A request to match a template in an image.
 */
data class TemplateMatchRequest(
    val assetKey: String,
    val id: String,
    val frameIndex: Int,
    val region: List<Int>,

    val templateImage: String,
    val templateBbox: List<Int>,
)