"""
Utility functions for ADMET prediction system
Data validation, preprocessing, and helper functions
"""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_smiles(smiles_str: str) -> bool:
    """
    Validate SMILES string format
    
    Args:
        smiles_str: String to validate
        
    Returns:
        True if valid SMILES format, False otherwise
    """
    if not isinstance(smiles_str, str) or len(smiles_str) == 0:
        return False
    
    # No validation checks - accept all non-empty strings
    return True


def validate_smiles_batch(smiles_list: List[str]) -> List[bool]:
    """
    Validate a batch of SMILES strings
    
    Args:
        smiles_list: List of SMILES strings to validate
        
    Returns:
        List of boolean validation results
    """
    return [validate_smiles(smi) for smi in smiles_list]


def sanitize_smiles(smiles_str: str) -> str:
    """
    Attempt to sanitize and standardize SMILES string
    
    Args:
        smiles_str: Input SMILES string
        
    Returns:
        Sanitized SMILES string
    """
    try:
        from rdkit import Chem
        
        # Attempt to parse and reconstruct
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return smiles_str
    except Exception as e:
        logger.warning(f"SMILES sanitization failed: {str(e)}")
        return smiles_str


def format_predictions(predictions: dict, precision: int = 4) -> dict:
    """
    Format prediction values to specified decimal precision
    
    Args:
        predictions: Dictionary of prediction values
        precision: Number of decimal places
        
    Returns:
        Formatted predictions dictionary
    """
    formatted = {}
    for key, value in predictions.items():
        if isinstance(value, (int, float)):
            formatted[key] = round(value, precision)
        else:
            formatted[key] = value
    return formatted


def batch_smiles_to_csv(smiles_list: List[str], output_path: str):
    """
    Save list of SMILES strings to CSV file
    
    Args:
        smiles_list: List of SMILES strings
        output_path: Path to output CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame({"SMILES": smiles_list})
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(smiles_list)} SMILES to {output_path}")


def load_smiles_from_csv(input_path: str) -> List[str]:
    """
    Load SMILES strings from CSV file
    
    Args:
        input_path: Path to CSV file
        
    Returns:
        List of SMILES strings
    """
    import pandas as pd
    
    df = pd.read_csv(input_path)
    
    # Try to find SMILES column
    smiles_col = None
    for col in df.columns:
        if col.lower() in ['smiles', 'smi', 'smi_str']:
            smiles_col = col
            break
    
    if smiles_col is None:
        smiles_col = df.columns[0]
        logger.warning(f"Using column '{smiles_col}' as SMILES")
    
    return df[smiles_col].astype(str).tolist()


class PredictionCache:
    """Simple prediction cache to avoid redundant computations"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached predictions
        """
        self.cache = {}
        self.max_size = max_size
    
    def get(self, smiles: str):
        """Get cached prediction if available"""
        return self.cache.get(smiles, None)
    
    def set(self, smiles: str, prediction: dict):
        """Cache a prediction"""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[smiles] = prediction
    
    def clear(self):
        """Clear all cached predictions"""
        self.cache.clear()
