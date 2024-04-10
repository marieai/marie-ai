package co.marieai

import co.marieai.client.TemplateMatcherClient
import co.marieai.model.BBox
import co.marieai.model.TemplateMatchingRequest
import co.marieai.model.TemplateSelector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import kotlin.random.Random

val charPool: List<Char> = ('a'..'z') + ('A'..'Z') + ('0'..'9')
fun generateRequestId() = (1..16)
    .map { Random.nextInt(0, charPool.size).let { charPool[it] } }
    .joinToString("")

fun toAbsolutePath(filename: String): Path {
    val relativePath: Path = Paths.get(filename)
    return  relativePath.toAbsolutePath().normalize()
}

private fun createRequest(): TemplateMatchingRequest {
    val filePath = "../../assets/template_matching/template-005_w.png"
    val bytes = Files.readAllBytes(Paths.get(filePath))
    val encoded = Base64.getEncoder().encodeToString(bytes)

    val template = TemplateSelector(
        region = BBox(0, 0, 0, 0),
        frame = encoded,
        bbox = BBox(174, 91, 91, 31),
        label = "Test",
        text = "",
        createWindow = true
    )

    val request = TemplateMatchingRequest(
        assetKey = toAbsolutePath("../../assets/template_matching/sample-005.png").toString(),
        id = generateRequestId(),
        pages = listOf(-1),
        scoreThreshold = 0.90,
        maxOverlap = 0.2,
        windowSize = Pair(512, 512),
        downscaleFactor = 0.0,
        scoringStrategy = "default",
        matcher = "composite",
        selectors = listOf(template)
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