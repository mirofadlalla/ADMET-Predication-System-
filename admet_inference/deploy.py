"""
ADMET Prediction System - Deployment Script
Automated setup for containerized inference deployment
Includes model checking and training from scratch if needed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def run_command(cmd, description):
    """Execute system command with error handling"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(f"✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  {e.stderr}")
        return False


def check_docker():
    """Verify Docker installation"""
    print_header("Checking Docker Installation")
    
    # Check Docker
    if run_command("docker --version", "Checking Docker"):
        print("✓ Docker found")
    else:
        print("✗ Docker not found. Please install Docker from https://www.docker.com")
        return False
    
    # Check Docker Compose
    if run_command("docker-compose --version", "Checking Docker Compose"):
        print("✓ Docker Compose found")
    else:
        print("✗ Docker Compose not found")
        return False
    
    return True


def check_models_exist():
    """Check if all required trained models exist"""
    print_header("Checking for Trained Models")
    
    models_dir = Path("./models")
    required_tasks = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    
    models_found = {}
    all_present = True
    
    for task in required_tasks:
        model_path = models_dir / task / "best_model.ckpt"
        exists = model_path.exists()
        models_found[task] = exists
        
        status = "✓ Found" if exists else "✗ Missing"
        print(f"  {task:<20} {status}")
        
        if not exists:
            all_present = False
    
    print()
    
    if all_present:
        print("✓ All models found! Ready for deployment")
        return True
    else:
        print("✗ Some models are missing")
        return False


def get_model_status():
    """Get count of existing models"""
    models_dir = Path("./models")
    required_tasks = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    
    count = 0
    for task in required_tasks:
        if (models_dir / task / "best_model.ckpt").exists():
            count += 1
    
    return count, len(required_tasks)


def train_models():
    """
    Train models from scratch
    Uses the ADMET_Professional.ipynb notebook for training
    """
    print_header("Training Models from Scratch")
    
    print("""
This requires running the ADMET_Professional.ipynb notebook.

To train models:
1. Ensure you have Jupyter installed: pip install jupyter
2. Navigate to parent directory: cd ..
3. Run Jupyter: jupyter notebook ADMET_Professional.ipynb
4. Execute all cells in order
5. Ensure models are saved to the 'trained_admet_models' directory
6. Copy models to: admet_inference/models/

Or use the provided training script below:
    """)
    
    # Check if we can run training directly
    try:
        import pandas as pd
        import torch
        print("✓ PyTorch and Pandas available - can train models")
        print("\nWould you like to run training now?")
        response = input("Enter 'yes' to train, or 'no' to skip: ").strip().lower()
        
        if response == 'yes':
            return run_training_direct()
        else:
            print("\nSkipping training. Please train models manually and then deploy.")
            return False
    except ImportError as e:
        print(f"⚠ Required package missing for direct training: {e}")
        print("\nPlease follow the manual training steps above.")
        return False


def run_training_direct():
    """
    Attempt to run training directly using Python
    """
    print("\nInitializing training environment...")
    
    try:
        # Import required libraries for training
        print("→ Importing training libraries...")
        import pandas as pd
        import numpy as np
        import torch
        from pathlib import Path
        
        print("✓ Libraries imported")
        
        # Check if TDC and ChemProp are available
        try:
            from tdc.single_pred import ADME, Tox
            print("✓ TDC library available")
        except ImportError:
            print("✗ TDC library required: pip install PyTDC")
            return False
        
        try:
            from chemprop import data, featurizers, models, nn
            from lightning import pytorch as pl
            from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
            from lightning.pytorch.loggers import MLFlowLogger
            print("✓ ChemProp and Lightning available")
        except ImportError:
            print("✗ ChemProp/Lightning required: pip install chemprop lightning")
            return False
        
        # Create models directory if it doesn't exist
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        
        print_header("Downloading and Preprocessing Data")
        
        # Define tasks
        tasks_config = {
            "Absorption": ("ADME", "Caco2_Wang"),
            "Distribution": ("ADME", "BBB_Martins"),
            "Metabolism": ("ADME", "CYP2D6_Veith"),
            "Excretion": ("ADME", "Half_Life_Obach"),
            "Toxicity": ("Tox", "hERG")
        }
        
        datasets = {}
        
        for task_name, (task_type, dataset_name) in tasks_config.items():
            try:
                print(f"\nDownloading {task_name}...")
                
                if task_type == "ADME":
                    data_obj = ADME(name=dataset_name)
                else:
                    data_obj = Tox(name=dataset_name)
                
                df = data_obj.get_data()
                
                # Standardize column names
                if 'Drug' in df.columns:
                    df = df.rename(columns={'Drug': 'SMILES'})
                elif 'Molecule' in df.columns:
                    df = df.rename(columns={'Molecule': 'SMILES'})
                
                if 'Y' in df.columns:
                    df = df.rename(columns={'Y': task_name})
                
                df = df[['SMILES', task_name]].dropna()
                
                print(f"  ✓ {task_name}: {len(df)} samples")
                datasets[task_name] = df
                
            except Exception as e:
                print(f"  ✗ Failed to download {task_name}: {str(e)}")
        
        if not datasets:
            print("\n✗ No datasets could be downloaded")
            return False
        
        print_header("Training Models")
        
        # Training configuration
        training_config = {
            'epochs': 50,
            'batch_size': 32,
            'patience': 8,
            'gradient_clip': 1.0
        }
        
        trained_count = 0
        
        for task_name, df in datasets.items():
            try:
                print(f"\nTraining {task_name}... (this may take several minutes)")
                
                # Prepare data
                smiles_list = df['SMILES'].values
                targets = df[[task_name]].values
                
                # Create datapoints
                datapoints = [
                    data.MoleculeDatapoint.from_smi(smi, y)
                    for smi, y in zip(smiles_list, targets)
                ]
                
                moles = [dp.mol for dp in datapoints if dp.mol is not None]
                
                if len(moles) == 0:
                    print(f"  ✗ No valid molecules in {task_name}")
                    continue
                
                # Split data
                train_idx, val_idx, test_idx = data.make_split_indices(
                    moles, 'scaffold_balanced', (0.8, 0.1, 0.1)
                )
                
                train_data, val_data, test_data = data.split_data_by_indices(
                    datapoints, train_idx, val_idx, test_idx
                )
                
                # Featurization
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                
                train_dset = data.MoleculeDataset(train_data[0], featurizer)
                scaler = train_dset.normalize_targets()
                
                val_dset = data.MoleculeDataset(val_data[0], featurizer)
                val_dset.normalize_targets(scaler)
                
                # DataLoaders
                train_loader = data.build_dataloader(
                    train_dset, batch_size=training_config['batch_size'], shuffle=True
                )
                val_loader = data.build_dataloader(
                    val_dset, batch_size=training_config['batch_size'], shuffle=False
                )
                
                # Model architecture
                mp = nn.BondMessagePassing()
                agg = nn.MeanAggregation()
                output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
                ffn = nn.RegressionFFN(output_transform=output_transform)
                
                model = models.MPNN(
                    mp, agg, ffn,
                    batch_norm=True,
                    metrics=[nn.metrics.RMSE(), nn.metrics.MAE()]
                )
                
                # Callbacks
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=training_config['patience'],
                    mode='min',
                    min_delta=0.001
                )
                
                task_model_dir = models_dir / task_name
                task_model_dir.mkdir(exist_ok=True)
                
                checkpoint = ModelCheckpoint(
                    dirpath=str(task_model_dir),
                    filename='best_model',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
                
                # Logger
                logger = MLFlowLogger(
                    experiment_name=f"ADMET_{task_name}",
                    tracking_uri="file:./logs"
                )
                
                # Training
                trainer = pl.Trainer(
                    logger=logger,
                    callbacks=[checkpoint, early_stop],
                    accelerator='auto',
                    devices=1,
                    max_epochs=training_config['epochs'],
                    gradient_clip_val=training_config['gradient_clip'],
                    log_every_n_steps=5,
                    enable_progress_bar=True
                )
                
                trainer.fit(model, train_loader, val_loader)
                print(f"  ✓ {task_name} training complete")
                trained_count += 1
                
            except Exception as e:
                print(f"  ✗ Training failed for {task_name}: {str(e)}")
        
        print_header(f"Training Summary: {trained_count}/{len(datasets)} models")
        
        if trained_count > 0:
            print(f"✓ Successfully trained {trained_count} models!")
            return True
        else:
            print("✗ No models were trained successfully")
            return False
        
    except ImportError as e:
        print(f"\n✗ Missing required libraries: {e}")
        print("\nPlease install required packages:")
        print("  pip install PyTDC chemprop lightning pandas numpy torch")
        return False
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        return False


def build_docker_image(image_name="admet-inference:latest", dockerfile_path="."):
    """Build Docker image"""
    print_header(f"Building Docker Image: {image_name}")
    
    return run_command(
        f"docker build -t {image_name} {dockerfile_path}",
        f"Building image {image_name}"
    )


def start_containers():
    """Start services with Docker Compose"""
    print_header("Starting Services")
    
    if run_command("docker-compose up -d", "Starting services"):
        print("\n✓ Services started successfully")
        print("\nAccess the API at:")
        print("  - Web Interface: http://localhost:8000/docs")
        print("  - Direct API: http://localhost:8000")
        return True
    return False


def stop_containers():
    """Stop services"""
    print_header("Stopping Services")
    return run_command("docker-compose down", "Stopping services")


def verify_services():
    """Verify all services are running"""
    print_header("Verifying Services")
    
    import time
    import requests
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✓ API is healthy and responding")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}: Waiting for API to start...")
                time.sleep(3)
            else:
                print("✗ API is not responding. Check logs with: docker-compose logs")
                return False
    
    return False


def view_logs():
    """Display service logs"""
    print_header("Service Logs")
    os.system("docker-compose logs -f admet-api")


def test_api():
    """Test API with sample prediction"""
    print_header("Testing API")
    
    import requests
    import json
    
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"smiles": test_smiles},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Test prediction successful!")
            print(f"\nInput SMILES: {test_smiles}")
            print(f"\nPredictions:")
            
            for task, value in result.get("predictions", {}).items():
                status = result.get("status", {}).get(task, "N/A")
                print(f"  {task:<15} {value:>7.4f}  ({status})")
            
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False


def show_menu():
    """Display interactive menu"""
    while True:
        print("\n" + "="*80)
        print(" ADMET Inference Deployment - Main Menu")
        print("="*80)
        print("\n1. Build Docker image")
        print("2. Start services")
        print("3. Stop services")
        print("4. View logs")
        print("5. Test API")
        print("6. Check models")
        print("7. Train models from scratch")
        print("8. Full setup (check → train if needed → build → start)")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            build_docker_image()
        elif choice == "2":
            start_containers()
            verify_services()
        elif choice == "3":
            stop_containers()
        elif choice == "4":
            view_logs()
        elif choice == "5":
            test_api()
        elif choice == "6":
            check_models_exist()
        elif choice == "7":
            train_models()
        elif choice == "8":
            print("\n" + "="*80)
            print(" FULL SETUP WORKFLOW: Check → Train (if needed) → Build → Start")
            print("="*80 + "\n")
            
            # Check models
            if check_models_exist():
                print("✓ Models already available, skipping training\n")
            else:
                print("\n⚠ Models not found. Starting training...\n")
                if not train_models():
                    print("\n⚠ Training skipped or failed. Continuing with deployment...")
                    print("⚠ Note: Inference will not work without trained models!")
            
            # Build and start
            if build_docker_image():
                if start_containers():
                    verify_services()
                    test_api()
        elif choice == "9":
            print("\n✓ Exiting...")
            break
        else:
            print("Invalid option. Please try again.")


def main():
    """Main deployment workflow"""
    print_header("ADMET Inference System - Deployment")
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deploy ADMET inference system with Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy.py --full              # Complete setup (check → train if needed → build → start)
  python deploy.py --check-models      # Check if models exist
  python deploy.py --train             # Train models from scratch
  python deploy.py --build --start     # Build and start (assumes models exist)
  python deploy.py -i                  # Interactive menu
        """
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Check if trained models exist"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models from scratch"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build Docker image"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start services"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop services"
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="View service logs"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full setup: check models → train if needed → build → start → verify → test"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive menu mode"
    )
    
    args = parser.parse_args()
    
    # Check Docker
    if not check_docker():
        sys.exit(1)
    
    # Execute requested operations
    if args.full:
        # Full workflow: check → train if needed → build → start → verify → test
        print("\n" + "="*80)
        print(" FULL DEPLOYMENT WORKFLOW")
        print("="*80)
        
        # Step 1: Check models
        print("\nStep 1: Checking for existing models...")
        models_exist = check_models_exist()
        
        # Step 2: Train if needed
        if not models_exist:
            print("\nStep 2: Models not found. Training from scratch...")
            if not train_models():
                print("\n⚠ Training failed. Continuing with deployment...")
                print("⚠ Note: API will not work without trained models!")
        else:
            print("\nStep 2: Models found. Skipping training.")
        
        # Step 3: Build
        print("\nStep 3: Building Docker image...")
        if not build_docker_image():
            print("✗ Build failed. Aborting deployment.")
            sys.exit(1)
        
        # Step 4: Start
        print("\nStep 4: Starting services...")
        if not start_containers():
            print("✗ Failed to start services. Aborting.")
            sys.exit(1)
        
        # Step 5: Verify
        print("\nStep 5: Verifying services...")
        if not verify_services():
            print("⚠ Services verification failed. Check logs with: docker-compose logs")
        
        # Step 6: Test
        print("\nStep 6: Testing API...")
        test_api()
        
        print_header("✓ FULL DEPLOYMENT COMPLETE")
    
    elif args.check_models:
        check_models_exist()
    
    elif args.train:
        train_models()
    
    elif args.build:
        build_docker_image()
    
    elif args.start:
        # Check models before starting
        if not check_models_exist():
            print("\n⚠ Warning: Models not found!")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Aborting. Please train models first with: python deploy.py --train")
                sys.exit(1)
        
        start_containers()
        verify_services()
    
    elif args.stop:
        stop_containers()
    
    elif args.logs:
        view_logs()
    
    elif args.test:
        test_api()
    
    elif args.interactive or len(sys.argv) == 1:
        show_menu()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
