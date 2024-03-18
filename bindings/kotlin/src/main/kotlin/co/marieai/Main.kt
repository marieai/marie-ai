package co.marieai

import co.marieai.client.MarieClient
import co.marieai.client.TemplateMatcherClient
import co.marieai.model.TemplateMatchRequest
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URI
import kotlin.random.Random

val charPool: List<Char> = ('a'..'z') + ('A'..'Z') + ('0'..'9')
fun generateRequestId() = (1..16)
    .map { Random.nextInt(0, charPool.size).let { charPool[it] } }
    .joinToString("")

fun getClient(url: URI): MarieClient {
    val channelBuilder = ManagedChannelBuilder.forAddress(url.host, url.port)
        // In case you are sending large data like images
        .maxInboundMessageSize(1024 * 1024 * 1024)
    if (url.scheme == "grpc") {
        channelBuilder.usePlaintext()
    }
    return MarieClient(channelBuilder.build())
}

suspend fun main() {
    val client = TemplateMatcherClient(getClient(URI("grpc://0.0.0.0:52000")))

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

    val request = TemplateMatchRequest("asset_key", "id", 0)
    val results = client.match(request)

    println("***********")
    println(results)
}