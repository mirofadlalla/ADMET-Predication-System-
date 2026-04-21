"""
ADMET Inference API - FastAPI Application
Production-ready REST API for ADMET inference (CPU-optimized, Async only)
Returns raw model predictions without interpretation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import time

from .inference import ADMETPredictor
from .utils import validate_smiles_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ADMET Inference System",
    description="REST API for drug ADMET property predictions (Async, Raw Outputs, CPU Optimized)",
    version="3.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = ADMETPredictor()
    logger.info("✓ ADMET Predictor initialized successfully")
    logger.info("✓ Mode: Async Inference Only (Raw Outputs)")
except Exception as e:
    logger.error(f"✗ Failed to initialize predictor: {str(e)}")
    predictor = None


# Request/Response Models
class PredictionRequest(BaseModel):
    """Single molecule prediction request"""
    smiles: str = Field(..., description="SMILES string representing the molecule")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    smiles_list: List[str] = Field(..., description="List of SMILES strings")


class PredictionResponse(BaseModel):
    """Single prediction response - Raw model outputs only"""
    smiles: str
    predictions: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total: int
    successful: int
    failed: int
    results: List[PredictionResponse]
    processing_time_ms: float



# Health & Status Endpoints
@app.get("/", tags=["System"])
async def root():
    """API information and documentation"""
    return {
        "name": "ADMET Inference System",
        "version": "3.0.0",
        "description": "REST API for drug ADMET property predictions (Async, Raw Outputs, CPU Optimized)",
        "mode": "inference-only-async",
        "output_type": "raw-predictions-only",
        "endpoints": {
            "docs": "/docs (Swagger UI)",
            "health": "/health",
            "predict": "/predict (Single async prediction)",
            "predict_batch": "/predict_batch (Batch async predictions)",
            "model_status": "/models/status"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Check system health and model availability"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    model_status = predictor.get_model_status()
    loaded_count = sum(1 for v in model_status.values() if v)
    
    return {
        "status": "healthy",
        "models_loaded": loaded_count,
        "total_models": len(model_status),
        "version": "3.0.0",
        "mode": "inference-only",
        "async": True,
        "output": "raw-predictions"
    }


@app.get("/models/status", tags=["System"])
async def model_status():
    """Get status of loaded models"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    model_status_dict = predictor.get_model_status()
    
    return {
        "models_loaded": model_status_dict,
        "total_models": len(model_status_dict),
        "models_ready": sum(1 for v in model_status_dict.values() if v),
        "admet_tasks": list(predictor.tasks)
    }



# Async Prediction Endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Async predict ADMET properties for a single molecule
    Returns raw model predictions without interpretation
    """
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    # Validate SMILES
    if not validate_smiles_batch([request.smiles])[0]:
        return PredictionResponse(
            smiles=request.smiles,
            predictions=None,
            error="Invalid SMILES string format"
        )
    
    try:
        result = await predictor.predict(request.smiles)
        
        return PredictionResponse(
            smiles=request.smiles,
            predictions=result['predictions'],
            error=None
        )
    
    except Exception as e:
        logger.error(f"Prediction error for {request.smiles}: {str(e)}")
        return PredictionResponse(
            smiles=request.smiles,
            predictions=None,
            error=str(e)
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Async predict ADMET properties for multiple molecules in parallel
    All processing is async for maximum performance
    """
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    start_time = time.time()
    
    # Validate batch
    valid_smiles = validate_smiles_batch(request.smiles_list)
    
    results = []
    successful = 0
    
    try:
        # Async batch processing - all molecules in parallel
        logger.info(f"Processing {len(request.smiles_list)} molecules async")
        async_results = await predictor.predict_batch(
            [smi for smi, valid in zip(request.smiles_list, valid_smiles) if valid]
        )
        
        result_idx = 0
        for smiles, is_valid in zip(request.smiles_list, valid_smiles):
            if not is_valid:
                results.append(PredictionResponse(
                    smiles=smiles,
                    predictions=None,
                    error="Invalid SMILES format"
                ))
                continue
            
            try:
                result = async_results[result_idx]
                if "error" in result:
                    results.append(PredictionResponse(
                        smiles=smiles,
                        predictions=None,
                        error=result["error"]
                    ))
                else:
                    results.append(PredictionResponse(
                        smiles=smiles,
                        predictions=result.get('predictions'),
                        error=None
                    ))
                    successful += 1
            except Exception as e:
                logger.error(f"Result processing error: {str(e)}")
                results.append(PredictionResponse(
                    smiles=smiles,
                    predictions=None,
                    error=str(e)
                ))
            
            result_idx += 1
    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        total=len(request.smiles_list),
        successful=successful,
        failed=len(request.smiles_list) - successful,
        results=results,
        processing_time_ms=processing_time
    )


# Legacy endpoint for backward compatibility
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch_legacy(request: BatchPredictionRequest):
    """Legacy endpoint - use /predict_batch instead"""
    return await predict_batch(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
