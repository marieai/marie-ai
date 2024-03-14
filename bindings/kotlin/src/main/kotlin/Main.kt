import io.grpc.ManagedChannelBuilder
import java.net.URI
import kotlin.random.Random

val charPool : List<Char> = ('a'..'z') + ('A'..'Z') + ('0'..'9')
fun generateRequestId() = (1..16)
    .map { Random.nextInt(0, charPool.size).let { charPool[it] } }
    .joinToString("")

fun getClient(url: URI): JinaClient {
    val channelBuilder = ManagedChannelBuilder.forAddress(url.host, url.port)
        // In case you are sending large data like images
        .maxInboundMessageSize(1024 * 1024 * 1024)
    if (url.scheme == "grpc") {
        channelBuilder.usePlaintext()
    }
    return JinaClient(channelBuilder.build())
}

suspend fun main() {
    val jinaClient = getClient(URI("grpcs://mareieai.co"))
    jinaClient.testReq1()
}