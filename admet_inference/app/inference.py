"""
Core async inference module for ADMET predictions - CPU-Optimized Inference Only
Handles model loading and concurrent prediction execution
Returns raw model outputs without interpretation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import asyncio

try:
    from chemprop import data, featurizers, models
except ImportError:
    raise ImportError("ChemProp library required. Install with: pip install chemprop")

logger = logging.getLogger(__name__)


class ADMETPredictor:
    """Unified ADMET property predictor using trained MPNN models - Inference Only (Async)"""
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize ADMET predictor with trained models
        
        Args:
            model_dir: Path to directory containing trained model checkpoints
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")  # CPU-only mode
        
        self.tasks = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
        self.models = {}
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all trained MPNN models from disk"""
        logger.info("Loading models...")
        for task in self.tasks:
            model_path = self.model_dir / task / "best_model.ckpt"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            
            try:
                model = models.MPNN.load_from_checkpoint(str(model_path))
                model.eval()
                model = model.to(self.device)
                self.models[task] = model
                logger.info(f"✓ Loaded {task} model")
            except Exception as e:
                logger.error(f"Failed to load {task} model: {str(e)}")
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {task: task in self.models for task in self.tasks}
    
    def _predict_sync(self, smiles: str) -> Dict:
        """
        Synchronous predict - used by async executor
        Returns only raw model predictions
        """
        try:
            # Create molecular datapoint
            datapoint = data.MoleculeDatapoint.from_smi(smiles)
            dataset = data.MoleculeDataset([datapoint], self.featurizer)
            loader = data.build_dataloader(dataset, batch_size=1, shuffle=False)
            
            predictions = {}
            
            # Generate predictions for each task
            with torch.inference_mode():
                for batch in loader:
                    for task in self.tasks:
                        if task not in self.models:
                            predictions[task] = None
                            continue
                        
                        model = self.models[task]
                        batch = batch[0].to(self.device), batch[1]
                        
                        output = model(batch[0], batch[1])
                        predictions[task] = float(output.cpu().numpy()[0][0])
            
            return {
                "smiles": smiles,
                "predictions": predictions
            }
        
        except Exception as e:
            logger.error(f"Prediction error for {smiles}: {str(e)}")
            raise
    
    async def predict(self, smiles: str) -> Dict:
        """
        Async predict ADMET properties for a single molecule
        Returns only raw model predictions (no interpretation)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, smiles)
    
    async def predict_batch(self, smiles_list: List[str]) -> List[Dict]:
        """
        Asynchronously predict ADMET properties for multiple molecules in parallel
        Returns only raw model predictions (no interpretation)
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of prediction dictionaries with raw model outputs
        """
        tasks = [self.predict(smiles) for smiles in smiles_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for smiles, result in zip(smiles_list, results):
            if isinstance(result, Exception):
                logger.error(f"Prediction failed for {smiles}: {str(result)}")
                processed_results.append({"smiles": smiles, "error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
