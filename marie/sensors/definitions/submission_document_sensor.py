"""Admit registered borrowed submission items without creating authority from S3 keys."""

from marie.sensors.definitions.kb_document_sensor import KbDocumentSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorType


class SubmissionSensorType:
    SUBMISSION_DOCUMENT = 'submission_document'


@register_sensor(SensorType.DATA_SINK, subtype='submission_document')
class SubmissionDocumentSensor(KbDocumentSensor):
    owner_type = 'submission_document'
