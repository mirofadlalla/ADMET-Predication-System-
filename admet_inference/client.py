"""
ADMET API Client Example
Demonstrates how to use the ADMET prediction API
"""

import requests
import pandas as pd
from typing import List, Dict
import time


class ADMETClient:
    """Client for ADMET prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def predict(self, smiles: str) -> Dict:
        """
        Predict ADMET properties for a single molecule
        
        Args:
            smiles: SMILES string
            
        Returns:
            Prediction dictionary
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"smiles": smiles},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {str(e)}")
            return None
    
    def predict_batch(self, smiles_list: List[str]) -> Dict:
        """
        Predict ADMET properties for multiple molecules
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Batch prediction results
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"smiles_list": smiles_list},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Batch prediction failed: {str(e)}")
            return None
    
    def predict_from_file(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict ADMET properties from CSV file
        
        Args:
            csv_path: Path to CSV file with SMILES column
            output_path: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        # Read input file
        df = pd.read_csv(csv_path)
        
        # Find SMILES column
        smiles_col = None
        for col in df.columns:
            if col.lower() in ['smiles', 'smi', 'smi_str']:
                smiles_col = col
                break
        
        if smiles_col is None:
            smiles_col = df.columns[0]
            print(f"Using column '{smiles_col}' as SMILES")
        
        smiles_list = df[smiles_col].tolist()
        
        # Make predictions
        print(f"Processing {len(smiles_list)} molecules...")
        results = self.predict_batch(smiles_list)
        
        if results is None:
            return None
        
        # Parse results
        output_df = pd.DataFrame()
        output_df['SMILES'] = [r['smiles'] for r in results['results']]
        
        # Extract predictions
        for task in ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']:
            output_df[task] = [
                r['predictions'].get(task) 
                if r.get('predictions') else None 
                for r in results['results']
            ]
            
            # Add status
            output_df[f"{task}_Status"] = [
                r['status'].get(task) 
                if r.get('status') else None 
                for r in results['results']
            ]
        
        # Save if output path provided
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return output_df


# Example usage
def main():
    """Example: Use ADMET API client"""
    
    print("ADMET API Client Example\n")
    
    # Initialize client
    client = ADMETClient()
    
    # Check health
    print("1. Health Check")
    if client.health_check():
        print("✓ API is healthy\n")
    else:
        print("✗ API is not responding\n")
        return
    
    # Single prediction
    print("2. Single Molecule Prediction")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    result = client.predict(smiles)
    
    if result:
        print(f"SMILES: {result['smiles']}")
        print(f"Valid: {result['valid']}")
        print("\nPredictions:")
        for task, value in result['predictions'].items():
            status = result['status'][task]
            print(f"  {task:<15} {value:>7.4f}  ({status})")
    print()
    
    # Batch prediction
    print("3. Batch Prediction")
    test_molecules = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    ]
    
    smiles_list = [smi for name, smi in test_molecules]
    names = [name for name, smi in test_molecules]
    
    batch_result = client.predict_batch(smiles_list)
    
    if batch_result:
        print(f"Total: {batch_result['total']}")
        print(f"Successful: {batch_result['successful']}")
        print(f"Failed: {batch_result['failed']}")
        print(f"Processing time: {batch_result['processing_time_ms']:.2f}ms\n")
        
        # Create results DataFrame
        results_df = pd.DataFrame()
        results_df['Compound'] = names
        
        for task in ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']:
            results_df[task] = [
                r['predictions'].get(task)
                if r.get('predictions') else None
                for r in batch_result['results']
            ]
        
        print("Results:")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('admet_predictions_example.csv', index=False)
        print("\n✓ Results saved to admet_predictions_example.csv")


if __name__ == "__main__":
    main()
