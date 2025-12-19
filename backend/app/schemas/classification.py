from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ClassificationRequest(BaseModel):
    text: str
    model_name: str
    file_name: Optional[str] = None


class ClassificationResponse(BaseModel):
    id: int
    text_content: str
    file_name: Optional[str]
    model_used: str
    prediction: str
    confidence: float
    created_at: datetime


class ModelInfo(BaseModel):
    name: str
    type: str  # "bert" or "llm"
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class LLMClassificationResult(BaseModel):
    result: str  # "real" or "fake"
    confidence: float  # 0.0 to 1.0


class LLMClassificationResponse(BaseModel):
    id: int
    text_content: str
    file_name: Optional[str]
    model_used: str
    classification: LLMClassificationResult
    created_at: datetime
