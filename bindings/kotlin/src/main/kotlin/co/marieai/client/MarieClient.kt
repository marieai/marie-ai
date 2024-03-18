package co.marieai.client

import com.google.protobuf.Empty
import io.grpc.ManagedChannel
import jina.Jina
import jina.Jina.StatusProto
import jina.JinaGatewayDryRunRPCGrpcKt
import jina.JinaSingleDataRequestRPCGrpcKt
import java.io.Closeable
import java.util.concurrent.TimeUnit
import io.grpc.Metadata

class MarieClient(
    private val channel: ManagedChannel
) : Closeable {
    //    private val stub: JinaRPCGrpcKt.JinaRPCCoroutineStub = JinaRPCGrpcKt.JinaRPCCoroutineStub(channel)
    private val stub: JinaSingleDataRequestRPCGrpcKt.JinaSingleDataRequestRPCCoroutineStub =
        JinaSingleDataRequestRPCGrpcKt.JinaSingleDataRequestRPCCoroutineStub(channel)

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
        try {
            val stub = JinaGatewayDryRunRPCGrpcKt.JinaGatewayDryRunRPCCoroutineStub(channel)
            val response = stub.dryRun(Empty.newBuilder().build(), Metadata())
            return true
        } catch (e: Exception) {
            return false
        }
    }
}