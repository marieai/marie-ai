from pydantic import BaseModel, ConfigDict


class TaxonomyPrediction(BaseModel):
    line_id: int
    label: str
    score: float
    model_config = ConfigDict(arbitrary_types_allowed=True)
