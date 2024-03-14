import io.grpc.ManagedChannelBuilder
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
    val client = getClient(URI("grpc://127.0.0.1:50001"))

    var maxWaitTime: Long = 10000
    var waitTime: Long = 1000
    var ready = false
    while (!(client.isReady().also { ready = it }) && (maxWaitTime > 0)) {
        println("Waiting for the server to start... " + maxWaitTime + "ms left.")
        Thread.sleep(waitTime)
        maxWaitTime -= waitTime
    }

    if (!ready) {
        println("Server is not up!")
        return
    }
    client.testReq1()
}