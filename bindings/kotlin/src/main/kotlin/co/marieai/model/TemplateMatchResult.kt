package co.marieai.model

/**
 * Represents a bounding box with its coordinates and dimensions.
 *
 * @property x The x-coordinate of the top-left corner of the bounding box.
 * @property y The y-coordinate of the top-left corner of the bounding box.
 * @property w The width of the bounding box.
 * @property h The height of the bounding box.
 */
data class BBox(val x: Int, val y: Int, val w: Int, val h: Int)

/**
 * Represents the result of template matching.
 *
 * @property bbox The bounding box coordinates of the matched template.
 * @property label The label of the matched template.
 * @property score The score of the template match.
 * @property similarity The similarity score of the template match.
 * @property frameIndex The index of the frame where the template was matched.
 */
data class TemplateMatchResult(
    val bbox: BBox,
    val label: String,
    val score: Float,
    val similarity: Float,
    val frameIndex: Int? = 0
)

/**
 * Represents a request to perform template matching.
 *
 * @property region The region of interest for template matching.
 * @property frame The image to use as a template for matching.(Base64 encoded)
 * @property bbox The bounding box coordinates of the template image.
 * @property label The label for the fragment
 * @property createWindow Whether to create a window for extract or use provider template image
 * @property topK Top K results per region
 */
data class TemplateSelector(
    val region: BBox,
    val frame: String,
    val bbox: BBox,
    val label: String,
    val text: String,
    val createWindow: Boolean,
    val topK: Int = 2,
)

data class TemplateMatchingRequest(
    val assetKey: String,
    var id: String,
    val pages: List<Int>,
    val scoreThreshold: Double = 0.90,
    val scoringStrategy: String = "weighted",
    val maxOverlap: Double = 0.5,
    val windowSize: Pair<Int, Int>,
    val downscaleFactor: Int,
    val matcher: String = "composite", // COMPOSITE, META, VQNNF
    val selectors: List<TemplateSelector>
)

/**
 * The `TemplateMatchingKeys` object provides a set of constant keys that are commonly used for template matching.
 * Keys here should match the pydantic model in Executor
 * This object should be used as a reference to access these constant keys in the template matching code.
 */
object TemplateMatchingKeys {
    const val ASSET_KEY = "asset_key"
    const val ID = "id"
    const val PAGES = "pages"
    const val SCORE_THRESHOLD = "score_threshold"
    const val SCORING_STRATEGY = "scoring_strategy"
    const val MAX_OVERLAP = "max_overlap"
    const val WINDOW_SIZE = "window_size"
    const val DOWNSCALE_FACTOR = "downscale_factor"
    const val MATCHER = "matcher"
    const val SELECTORS = "selectors"
}

object TemplateSelectorKeys {
    const val REGION = "region"
    const val FRAME = "frame"
    const val BBOX = "bbox"
    const val LABEL = "label"
    const val TEXT = "text"
    const val CREATE_WINDOW = "create_window"
    const val TOP_K = "top_k"
}
