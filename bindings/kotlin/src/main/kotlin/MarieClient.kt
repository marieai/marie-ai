import com.google.protobuf.Empty
import com.google.protobuf.Struct
import docarray.*

import io.grpc.ManagedChannel
import jina.*
import kotlinx.coroutines.flow.asFlow
import java.io.Closeable
import java.util.concurrent.TimeUnit
import io.grpc.Metadata

class MarieClient(
    private val channel: ManagedChannel
) : Closeable {
    private val stub: JinaRPCGrpcKt.JinaRPCCoroutineStub = JinaRPCGrpcKt.JinaRPCCoroutineStub(channel)

    private val stubDryRun: JinaGatewayDryRunRPCGrpcKt.JinaGatewayDryRunRPCCoroutineStub =
        JinaGatewayDryRunRPCGrpcKt.JinaGatewayDryRunRPCCoroutineStub(channel)

    suspend fun testReq1() {

        val response = stubDryRun.dryRun(Empty.newBuilder().build(), Metadata())
        println(response)

        /*
        Original:
            c = Client(host='https://127.0.0.1')
            print(c.post('/', Document(text='hello')))
         */
        val dataReq = dataRequestProto {
            data = DataRequestProtoKt.dataContentProto {
                this.docs = documentArrayProto {
                    this.docs.add(documentProto {
                        this.text = "hello"
                        this.uri = "uri"
                    })
                }
            }
            header = headerProto {
                execEndpoint = "/document/extract"
                requestId = generateRequestId()
            }
        }
        val reqs = listOf(dataReq).asFlow()
        stub.withCompression("gzip").call(reqs).collect {
            // Get first doc as reply (Note! This is just an example, you can choose different approaches depending on you use-case).
            val doc = it.data.docs.getDocs(0)
            // Loop over list of docs
            for (doc in it.data.docs.docsList) {
                println("Doc: $doc");
            }
        }
    }


    override fun close() {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS)
    }

    /**
     * Check if the server is ready
     */
    suspend fun isReady(): Boolean {
        try {
            val response = stubDryRun.dryRun(Empty.newBuilder().build(), Metadata())
            println(response)
        } catch (e: Exception) {
            return false
        }
        return true
    }
}