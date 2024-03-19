package co.marieai

import co.marieai.client.MarieClient
import co.marieai.client.TemplateMatcherClient
import co.marieai.model.BBox
import co.marieai.model.TemplateSelector
import co.marieai.model.TemplateMatchingRequest
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URI
import kotlin.random.Random
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Base64

val charPool: List<Char> = ('a'..'z') + ('A'..'Z') + ('0'..'9')
fun generateRequestId() = (1..16)
    .map { Random.nextInt(0, charPool.size).let { charPool[it] } }
    .joinToString("")



private fun createRequest(): TemplateMatchingRequest {
    val filePath = "../../assets/template_matching/sample-005.png"
    val bytes = Files.readAllBytes(Paths.get(filePath))
    val encoded = Base64.getEncoder().encodeToString(bytes)

    val template = TemplateSelector(
        region = BBox(0, 0, 0, 0),
        frame = encoded,
        bbox = BBox(223, 643, 124, 38),
        label = "Test",
        text = "",
        createWindow = false
    )

    val request = TemplateMatchingRequest(
        assetKey = "DocumentId-001",
        id = generateRequestId(),
        pages = listOf(-1),
        scoreThreshold = 0.90,
        maxOverlap = 0.2,
        windowSize = Pair(512, 512),
        downscaleFactor = 1,
        scoringStrategy = "default",
        matcher = "composite",
        selectors = listOf(template)
    )

    return request
}

suspend fun main() {
    val client = TemplateMatcherClient( URI("grpc://127.0.0.1:52000") )

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

    println("***********")
    println(results)
}