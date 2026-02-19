package co.marieai

import co.marieai.client.TemplateMatcherClient
import co.marieai.model.BBox
import co.marieai.model.TemplateMatchingRequest
import co.marieai.model.TemplateSelector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URI
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*

private fun templateSelector(label: String, region: BBox, filePath: String): TemplateSelector {
    val bytes = Files.readAllBytes(Paths.get(filePath))
    val encoded = Base64.getEncoder().encodeToString(bytes)

    return TemplateSelector(
        region = region,
        frame = encoded,
        bbox = BBox(0, 0, 0, 0), // we are creating BBOX from create
        label = label,
        text = "",
        createWindow = true,
        topK = 2
    )
}


private fun createRequest(): TemplateMatchingRequest {
    val template0 = templateSelector("see_note", BBox(0, 0, 0, 0), "~/tmp/anchors/215219944.png")
    val template1 = templateSelector("codes", BBox(0, 0, 0, 0), "~/tmp/anchors/76432244.png")
    val template2 = templateSelector("totals", BBox(0, 0, 0, 0), "~/tmp/anchors/1099855928.png")

    val request = TemplateMatchingRequest(
        assetKey = "75160793.tif",// multipage
        id = generateRequestId(),
        pages = listOf(2, 3),
        scoreThreshold = 0.90,
        scoringStrategy = "weighted",
        maxOverlap = 0.2,
        windowSize = Pair(512, 512),
        downscaleFactor = 0.0,
        matcher = "composite",
        selectors = listOf(template0, template1, template2)
    )

    return request
}


suspend fun main() {
    val client = TemplateMatcherClient(URI("grpc://127.0.0.1:52000"))

    var maxWaitTime: Long = 10_000
    val waitTime: Long = 1000
    var ready: Boolean

    while (!(client.isReady().also { ready = it }) && (maxWaitTime > 0)) {
        println("Waiting for the server to start... " + maxWaitTime + "ms left.")
        withContext(Dispatchers.IO) {
            Thread.sleep(waitTime)
        }
        maxWaitTime -= waitTime
    }

    if (!ready) {
        println("Server is not up!")
        return
    }

    val request = createRequest()
    val results = client.match(request)

    print("Reply received: ")
    println("***********")
    println(results)
}