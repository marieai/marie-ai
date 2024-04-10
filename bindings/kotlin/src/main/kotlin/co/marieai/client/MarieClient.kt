package co.marieai.client

import co.marieai.model.BBox
import com.google.protobuf.Empty
import docarray.DocProtoKt
import docarray.Docarray
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import jina.Jina
import jina.Jina.StatusProto
import jina.JinaGatewayDryRunRPCGrpcKt
import jina.JinaSingleDataRequestRPCGrpcKt
import java.io.Closeable
import java.util.concurrent.TimeUnit
import io.grpc.Metadata
import java.net.URI

/***
 * MarieClient is the base class for all clients in the MarieAI SDK.
 * This class is responsible for creating a gRPC channel and handling the communication with the server.
 * As the namespoace of 'jina' have been kept the same, the class can be used to communicate with the Jina server as well.
 */
open class MarieClient(
    uri: URI
) : Closeable {
    private val channel: ManagedChannel = getChannel(uri)
    private val stub = JinaSingleDataRequestRPCGrpcKt.JinaSingleDataRequestRPCCoroutineStub(channel)

    private fun getChannel(url: URI): ManagedChannel {
        val channelBuilder = ManagedChannelBuilder.forAddress(url.host, url.port)
            // In case you are sending large data like images
            .maxInboundMessageSize(1024 * 1024 * 1024)
        if (url.scheme == "grpc") {
            channelBuilder.usePlaintext()
        }
        return channelBuilder.build()
    }

    protected fun <T> convertListToProto(
        items: List<T>,
        nodeBuilder: (T) -> Docarray.NodeProto
    ): Docarray.ListOfAnyProto? {
        val builder = Docarray.ListOfAnyProto.newBuilder()
        items.forEach { item -> builder.addData(nodeBuilder(item)) }
        return builder.build()
    }

    protected fun convertBBox(bbox: BBox): Docarray.NodeProto {
        return Docarray.NodeProto.newBuilder()
            .setList(
                convertListToProto(listOf(bbox.x, bbox.y, bbox.w, bbox.h)) {
                    Docarray.NodeProto.newBuilder().setInteger(it).build()
                })
            .build()
    }

    protected fun <T> DocProtoKt.Dsl.putData(
        builder: Docarray.NodeProto.Builder,
        key: String,
        value: T,
        settingFunction: Docarray.NodeProto.Builder.(T) -> Docarray.NodeProto.Builder
    ) {
        data.put(key, builder.settingFunction(value).build())
    }


    suspend fun processSingleData(dataReq: Jina.DataRequestProto): Jina.DataRequestProto {
        val response = stub.withCompression("gzip").processSingleData(dataReq)
        if (response.header.status.code != StatusProto.StatusCode.SUCCESS) {
            println("Error: ${response.header.status}")
            return response
        }
        return response
    }

    override fun close() {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS)
    }

    /**
     * Check if the server is ready
     */
    suspend fun isReady(): Boolean {
        return try {
            val stub = JinaGatewayDryRunRPCGrpcKt.JinaGatewayDryRunRPCCoroutineStub(channel)
            val response = stub.dryRun(Empty.newBuilder().build(), Metadata())
            true
        } catch (e: Exception) {
            false
        }
    }
}