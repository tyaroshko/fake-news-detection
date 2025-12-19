import asyncio
import logging
from typing import List, Union

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from ..models.database import ClassificationResult, SessionLocal, get_db
from ..schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
    LLMClassificationResponse,
    LLMClassificationResult,
    ModelInfo,
    ModelsResponse,
)
from ..services.classification_service import classification_service

router = APIRouter()


@router.post(
    "/classify/text",
    response_model=Union[ClassificationResponse, LLMClassificationResponse],
)
async def classify_text(request: ClassificationRequest, db: Session = Depends(get_db)):
    """Classify text content as real or fake news."""
    logger.info(f"Received text classification request for model: {request.model_name}")
    try:
        # Get model config to determine response type
        model_config = classification_service.available_models.get(request.model_name)
        if not model_config:
            raise HTTPException(
                status_code=400, detail=f"Model {request.model_name} not available"
            )

        # Run classification
        result = await classification_service.classify_text(
            request.text, request.model_name
        )

        # Handle different response types
        if isinstance(result, LLMClassificationResult):
            # LLM response
            db_result = ClassificationResult(
                text_content=request.text,
                file_name=request.file_name,
                model_used=request.model_name,
                prediction=result.result,
                confidence=result.confidence,
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)

            logger.info(
                f"LLM classification completed: {result.result} (confidence: {result.confidence:.3f})"
            )
            return LLMClassificationResponse(
                id=db_result.id,
                text_content=db_result.text_content,
                file_name=db_result.file_name,
                model_used=db_result.model_used,
                classification=result,
                created_at=db_result.created_at,
            )
        else:
            # BERT response (tuple)
            prediction, confidence = result
            db_result = ClassificationResult(
                text_content=request.text,
                file_name=request.file_name,
                model_used=request.model_name,
                prediction=prediction,
                confidence=confidence,
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)

            return ClassificationResponse(
                id=db_result.id,
                text_content=db_result.text_content,
                file_name=db_result.file_name,
                model_used=db_result.model_used,
                prediction=db_result.prediction,
                confidence=db_result.confidence,
                created_at=db_result.created_at,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post(
    "/classify/file",
    response_model=Union[ClassificationResponse, LLMClassificationResponse],
)
async def classify_file(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    db: Session = Depends(get_db),
):
    """Classify content from uploaded file."""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8")

        # Get model config to determine response type
        model_config = classification_service.available_models.get(model_name)
        if not model_config:
            raise HTTPException(
                status_code=400, detail=f"Model {model_name} not available"
            )

        # Run classification
        result = await classification_service.classify_text(text_content, model_name)

        # Handle different response types
        if isinstance(result, LLMClassificationResult):
            # LLM response
            db_result = ClassificationResult(
                text_content=text_content,
                file_name=file.filename,
                model_used=model_name,
                prediction=result.result,
                confidence=result.confidence,
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)

            return LLMClassificationResponse(
                id=db_result.id,
                text_content=db_result.text_content[:200],  # Truncate for response
                file_name=db_result.file_name,
                model_used=db_result.model_used,
                classification=result,
                created_at=db_result.created_at,
            )
        else:
            # BERT response (tuple)
            prediction, confidence = result
            db_result = ClassificationResult(
                text_content=text_content,
                file_name=file.filename,
                model_used=model_name,
                prediction=prediction,
                confidence=confidence,
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)

            return ClassificationResponse(
                id=db_result.id,
                text_content=db_result.text_content[:200],  # Truncate for response
                file_name=db_result.file_name,
                model_used=db_result.model_used,
                prediction=db_result.prediction,
                confidence=db_result.confidence,
                created_at=db_result.created_at,
            )

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a text file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """Get list of available classification models."""
    models = classification_service.get_available_models()
    return ModelsResponse(models=models)


@router.get(
    "/results",
    response_model=List[Union[ClassificationResponse, LLMClassificationResponse]],
)
async def get_classification_results(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """Get classification history."""
    results = db.query(ClassificationResult).offset(skip).limit(limit).all()
    response_list = []

    for result in results:
        # Check if this was an LLM model based on the model name
        model_config = classification_service.available_models.get(result.model_used)
        is_llm = (
            model_config and model_config["type"] == "llm" if model_config else False
        )

        if is_llm:
            # Return LLM response format
            llm_result = LLMClassificationResult(
                result=result.prediction, confidence=result.confidence
            )
            response_list.append(
                LLMClassificationResponse(
                    id=result.id,
                    text_content=result.text_content[:200],  # Truncate for response
                    file_name=result.file_name,
                    model_used=result.model_used,
                    classification=llm_result,
                    created_at=result.created_at,
                )
            )
        else:
            # Return BERT response format
            response_list.append(
                ClassificationResponse(
                    id=result.id,
                    text_content=result.text_content[:200],  # Truncate for response
                    file_name=result.file_name,
                    model_used=result.model_used,
                    prediction=result.prediction,
                    confidence=result.confidence,
                    created_at=result.created_at,
                )
            )

    return response_list
