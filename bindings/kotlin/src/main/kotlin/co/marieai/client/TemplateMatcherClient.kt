package co.marieai.client

import co.marieai.generateRequestId
import co.marieai.model.*
import docarray.DocProtoKt
import docarray.Docarray
import docarray.Docarray.ListOfAnyProto
import docarray.Docarray.NodeProto
import docarray.Docarray.NodeProto.Builder
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

    private fun <T> convertListToProto(items: List<T>, nodeBuilder: (T) -> NodeProto): ListOfAnyProto? {
        val builder = ListOfAnyProto.newBuilder()
        items.forEach { item -> builder.addData(nodeBuilder(item)) }
        return builder.build()
    }

    private fun convertBBox(bbox: BBox): NodeProto {
        return NodeProto.newBuilder()
            .setList(
                convertListToProto(listOf(bbox.x, bbox.y, bbox.w, bbox.h)) {
                    NodeProto.newBuilder().setInteger(it).build()
                })
            .build()
    }


    private fun convertSelectorToProto(selector: TemplateSelector): NodeProto {
        val builder = NodeProto.newBuilder()
        return builder.setDoc(
            docProto {
                data.putAll(
                    mapOf(
                        TemplateSelectorKeys.REGION to convertBBox(selector.region),
                        TemplateSelectorKeys.FRAME to builder.setText(selector.frame).build(),
                        TemplateSelectorKeys.BBOX to convertBBox(selector.bbox),
                        TemplateSelectorKeys.CREATE_WINDOW to builder.setBoolean(selector.createWindow)
                            .build(),
                        TemplateSelectorKeys.TEXT to builder.setText(selector.text).build(),
                        TemplateSelectorKeys.TOP_K to builder.setInteger(selector.topK).build(),
                        TemplateSelectorKeys.LABEL to builder.setText(selector.label).build()
                    )
                )
            }).build()
    }

    private fun convertSelectorsToProto(selectors: List<TemplateSelector>): ListOfAnyProto? {
        val protoListBuilder = ListOfAnyProto.newBuilder()
        for (selector in selectors) {
            var converted = convertSelectorToProto(selector)
            protoListBuilder.addData(converted)
        }
        return protoListBuilder.build()
    }

    private fun <T> DocProtoKt.Dsl.putData(
        builder: Builder,
        key: String,
        value: T,
        settingFunction: Builder.(T) -> Builder
    ) {
        data.put(key, builder.settingFunction(value).build())
    }


    private fun buildDocProto(request: TemplateMatchingRequest) = docProto {
        val builder = NodeProto.newBuilder()

        putData(builder, TemplateMatchingKeys.ID, request.id, Builder::setText)
        putData(builder, TemplateMatchingKeys.ASSET_KEY, request.assetKey, Builder::setText)
        putData(builder, TemplateMatchingKeys.PAGES, convertListToProto(request.pages) {
            NodeProto.newBuilder().setInteger(it).build()
        }, Builder::setList)
        putData(builder, TemplateMatchingKeys.SCORE_THRESHOLD, request.scoreThreshold, Builder::setFloat)
        putData(builder, TemplateMatchingKeys.SCORING_STRATEGY, request.scoringStrategy, Builder::setText)
        putData(builder, TemplateMatchingKeys.MAX_OVERLAP, request.maxOverlap, Builder::setFloat)
        putData(
            builder, TemplateMatchingKeys.WINDOW_SIZE, convertListToProto(
                listOf(
                    request.windowSize.first,
                    request.windowSize.second
                )
            ) {
                NodeProto.newBuilder().setInteger(it).build()
            }, Builder::setList
        )
        putData(builder, TemplateMatchingKeys.MATCHER, request.matcher, Builder::setText)
        putData(builder, TemplateMatchingKeys.DOWNSCALE_FACTOR, request.downscaleFactor, Builder::setInteger)
        putData(builder, TemplateMatchingKeys.SELECTORS, convertSelectorsToProto(request.selectors), Builder::setList)
    }

    suspend fun match(request: TemplateMatchingRequest): List<TemplateMatchResult> {
        val rid = generateRequestId()
        println("Generated request id: $rid")
        val builder = NodeProto.newBuilder()

        val dataReq = dataRequestProto {
            data = DataRequestProtoKt.dataContentProto {
                docs = docListProto {
                    docs.add(buildDocProto(request))
                }
            }
            header = headerProto {
                execEndpoint = "/document/matcher"
                requestId = rid
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

            val templateMatchResult =
                TemplateMatchResult(BBox(bbox[0], bbox[1], bbox[2], bbox[3]), label, score, similarity, frameIndex)
            templateMatchResults.add(templateMatchResult)
        }
        return templateMatchResults
    }

}