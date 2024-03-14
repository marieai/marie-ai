import docarray.documentArrayProto
import docarray.documentProto
import io.grpc.ManagedChannel
import jina.DataRequestProtoKt
import jina.JinaRPCGrpcKt
import jina.dataRequestProto
import jina.headerProto
import kotlinx.coroutines.flow.asFlow
import java.io.Closeable
import java.util.concurrent.TimeUnit

class JinaClient(
    private val channel: ManagedChannel
) : Closeable {
    private val stub: JinaRPCGrpcKt.JinaRPCCoroutineStub = JinaRPCGrpcKt.JinaRPCCoroutineStub(channel)

    suspend fun testReq1() {
        /*
        Original:
            c = Client(host='https://mariegateway.com')
            print(c.post('/', Document(text='hello')))
         */
        val dataReq = dataRequestProto {
            data = DataRequestProtoKt.dataContentProto {
                this.docs = documentArrayProto {
                    this.docs.add(documentProto {
                        this.text = "hello"
                    })
                }
            }
            header = headerProto {
                execEndpoint = "/"
                requestId = generateRequestId()
            }
        }
        val reqs = listOf(dataReq).asFlow()
        stub.withCompression("gzip").call(reqs).collect {
            // Get first doc as reply (Note! This is just an example, you can choose different approaches depending on you use-case).
            val doc = it.data.docs.getDocs(0)
            // Loop over list of docs
            for(doc in it.data.docs.docsList) {
                println("Doc: $doc");
            }
        }
    }

    override fun close() {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS)
    }
}