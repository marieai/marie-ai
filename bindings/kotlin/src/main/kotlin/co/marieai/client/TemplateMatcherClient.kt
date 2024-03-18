package co.marieai.client

import co.marieai.generateRequestId
import co.marieai.model.TemplateMatchResult
import co.marieai.model.TemplateMatchRequest
import docarray.Docarray
import docarray.docListProto
import docarray.docProto
import jina.DataRequestProtoKt
import jina.Jina.StatusProto
import jina.dataRequestProto
import jina.headerProto
import java.io.Closeable
import kotlin.collections.emptyList as emptyList1

class TemplateMatcherClient(
    private val client: MarieClient
) : Closeable {

    suspend fun match(request: TemplateMatchRequest): List<TemplateMatchResult> {
        val gid = generateRequestId()
        println("Generated request id: $gid")
        val dataReq = dataRequestProto {
            data = DataRequestProtoKt.dataContentProto {
                docs = docListProto {
                    docs.add(docProto {
                        data.put("asset_key", Docarray.NodeProto.newBuilder().setText("SOURCE_ASSET_KEY").build())
                        data.put("regions", Docarray.NodeProto.newBuilder().setText("RegionABCD").build())
                    })
                }
            }
            header = headerProto {
                execEndpoint = "/document/matcher"
                requestId = gid
            }
        }

        val response = client.processSingleData(dataReq)
        if (response.header.status.code != StatusProto.StatusCode.SUCCESS) {
            println("Error: ${response.header.status}")
            return emptyList1()
        }

        println("response ******** : \n")
        for (doc in response.data.docs.docsList) {
            println("\n-------------------\n")
            println("Doc: $doc");

            val result = toTemplateMatchResult(doc)
            println("Result: $result")
            return result
        }
        return emptyList1()
    }

    override fun close() {
        client.close()
    }

    /**
     * Check if the server is ready
     */
    suspend fun isReady(): Boolean {
        return client.isReady()
    }

    private fun toTemplateMatchResult(doc: Docarray.DocProto): List<TemplateMatchResult> {
        val dm = doc.dataMap
        val assetKey = dm["asset_key"]?.text ?: ""
        val id = dm["id"]?.text ?: ""
        val results = dm["results"]?.list

        println("-------------------\n")
        println("asset_key: $assetKey")
        println("id: $id")
        println(results)

        val templateMatchResults = mutableListOf<TemplateMatchResult>()

        for (result in results?.dataList!!) {
            val rdm = result.doc.dataMap
            val frameIndex = rdm["frame_index"]?.integer ?: 0
            val label = rdm["label"]?.text ?: ""
            val score = rdm["score"]?.float?.toFloat() ?: 0f
            val similarity = rdm["similarity"]?.float?.toFloat() ?: 0f
            val bbox = rdm["bbox"]?.tuple?.dataList?.map { it.integer } ?: emptyList1()

            println("frame_index: $frameIndex")
            println("label: $label")
            println("score: $score")
            println("similarity: $similarity")
            println("bbox: $bbox")

            val templateMatchResult = TemplateMatchResult(bbox, label, score, similarity, frameIndex)
            templateMatchResults.add(templateMatchResult)
        }
        return templateMatchResults
    }

}