"""
Model Validation Pipeline - For Testing Pre-trained DeepLOB Models

This module provides a comprehensive framework for validating pre-trained DeepLOB
(Deep Learning for Limit Order Book) models on specific time periods of cryptocurrency data.
It enables detailed performance analysis using multiple metrics, with a focus on 
directional prediction accuracy for financial market movements.

Key features:
- Loads pre-trained model weights for the DeepLOB architecture
- Processes normalized order book data from specified time periods
- Performs GPU-accelerated model validation using optimized data loaders
- Calculates specialized metrics for financial prediction evaluation
- Creates detailed visualizations and reports for model performance analysis
- Supports comparison of predicted values with actual price movements
"""
import os, datetime as dt, time
from pathlib import Path

# Data visualization and analysis libraries
import matplotlib.pyplot as plt  # For creating plots and visualizations
import seaborn as sns           # For enhanced statistical visualizations
import numpy as np              # For numerical operations and array manipulation
import pandas as pd             # For data manipulation and analysis

# Deep learning framework
import torch                    # PyTorch deep learning library
from torch import nn            # Neural network modules
from sklearn.metrics import f1_score, confusion_matrix, classification_report  # For evaluation metrics

# Import custom DeepLOB model and data loading utilities
from deeplob_optimized import (
    DeepLOB,                   # The deep learning model architecture for LOB data
    load_book_chunk            # Utility to load normalized LOB data chunks
)

# Import GPU-optimized dataset and data loader implementation
from gpu_loaders import create_gpu_data_loaders

# Configure PyTorch to use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Enable PyTorch performance optimizations for faster computation
torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for faster matrix operations
torch.backends.cudnn.allow_tf32 = True  # Enable TensorFloat-32 for cuDNN operations

class ModelValidator:
    """
    Comprehensive validator class for evaluating pre-trained DeepLOB models.
    
    This class handles the entire validation pipeline for a pre-trained DeepLOB model,
    including loading model weights, preparing validation data, performing inference,
    calculating performance metrics, and generating visualizations and reports.
    
    The validator focuses on directional prediction accuracy (up/down price movements)
    as the primary metric for financial trading applications, while also providing
    standard classification metrics for all three classes (down, stable, up).
    """
    def __init__(self, 
                 file_paths,
                 model_path,
                 depth=10,
                 window=100,
                 horizon=100,
                 batch_size=64,
                 alpha=0.002,
                 stride=1):
        """
        Initialize the model validation framework with configuration parameters.
        
        Args:
            file_paths: List of dictionaries containing file information for processing
                        Each dict contains 'path', 'filename', 'start_date', 'end_date'
            model_path: Path to the pre-trained model (.pt file) to evaluate
            depth: Number of price levels in the limit order book (LOB depth)
            window: Size of the time window for input features (number of time steps)
            horizon: Prediction horizon for future price movements (number of time steps ahead)
            batch_size: Batch size for validation inference
            alpha: Threshold value for classifying price changes (0=down, 1=stable, 2=up)
            stride: Step size for sampling data points (default: 1 for validation to include all points)
        """
        # Store configuration parameters as instance variables
        self.file_paths = file_paths
        self.model_path = model_path
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.alpha = alpha
        self.stride = stride
        
        # Verify that the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified model file could not be found: {model_path}")
        
        # Initialize the DeepLOB model architecture
        print(f"Initializing model from {model_path}...")
        self.model = DeepLOB(depth=depth).to(
            device, 
            memory_format=torch.channels_last  # Use channels-last memory format for better GPU utilization
        )
        
        # Load model weights with graceful fallback for compatibility
        try:
            # First try with the safer weights_only option (PyTorch >= 2.0)
            checkpoint = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f"Safe loading failed, falling back to full object loading: {e}")
            # If it fails, use the less secure but more compatible mode
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
        
        # Display original training performance metrics from the checkpoint
        print(f"Model loaded successfully. Original training metrics:")
        if 'directional_f1' in checkpoint:
            print(f"- Directional F1: {checkpoint['directional_f1']:.4f}")
        if 'f1' in checkpoint:
            print(f"- Global F1: {checkpoint['f1']:.4f}")
        if 'epoch' in checkpoint:
            print(f"- Trained for {checkpoint['epoch']} epochs")
        
        # Initialize data containers for validation
        print("\nLoading data for validation...")
        self.data = []  # Will store normalized data frames
        self.raw_prices = []  # Will store raw price data for comparison
        
        # Process each file in the provided file paths
        for file_info in self.file_paths:
            print(f"Reading normalized file: {file_info['filename']}")
            print(f"Time range: {file_info['start_date']} to {file_info['end_date']}")
            
            # Load normalized data from parquet file
            df = pd.read_parquet(file_info['path'])
            
            # Filter data to match the specified time range
            mask = (df['received_time'] >= file_info['start_date']) & \
                   (df['received_time'] <= file_info['end_date'])
            df = df[mask]
            
            # Check if any data exists for the time period
            if len(df) > 0:
                self.data.append(df)
                print(f"Loaded {len(df)} normalized records")
                
                # Load corresponding raw data for price comparison
                # This helps verify model predictions against actual price movements
                raw_file_path = os.path.join(
                    'szakdolgozat-high-freq-btc-prediction/data_raw',
                    f"book_btc_usdt_{file_info['start_date'].strftime('%Y%m%d')}_{file_info['end_date'].strftime('%Y%m%d')}.parquet"
                )
                
                # Check if raw data file exists
                if os.path.exists(raw_file_path):
                    print(f"Reading raw file: {raw_file_path}")
                    raw_df = pd.read_parquet(raw_file_path)
                    
                    # Filter raw data to match the same time range
                    raw_mask = (raw_df['received_time'] >= file_info['start_date']) & \
                             (raw_df['received_time'] <= file_info['end_date'])
                    raw_df = raw_df[raw_mask]
                    
                    if len(raw_df) > 0:
                        # Extract only timestamp and best ask price columns for efficiency
                        raw_prices = raw_df[['received_time', 'ask_0_price']]
                        self.raw_prices.append(raw_prices)
                        print(f"Loaded {len(raw_prices)} raw price records")
                    else:
                        print(f"No raw data found in time range {file_info['start_date']} to {file_info['end_date']}")
                else:
                    print(f"Raw file not found: {raw_file_path}")
            else:
                print(f"No normalized data found in time range {file_info['start_date']} to {file_info['end_date']}")
        
        # Ensure we have data to validate with
        if not self.data:
            raise ValueError("No data found in the specified time ranges")
        
        # Concatenate all normalized data frames into a single dataframe for processing
        self.combined_data = pd.concat(self.data, ignore_index=True)
        print(f"\nTotal normalized data points loaded: {len(self.combined_data)}")
        
        # Concatenate raw price data if available
        if self.raw_prices:
            self.combined_raw_prices = pd.concat(self.raw_prices, ignore_index=True)
            print(f"Total raw price data points loaded: {len(self.combined_raw_prices)}")
        else:
            self.combined_raw_prices = None
            print("Warning: No raw price data loaded")
        
        # Display the full time range of the loaded data
        if len(self.combined_data) > 0:
            print(f"Time range: {self.combined_data['received_time'].min()} to {self.combined_data['received_time'].max()}")        
        # Create a GPU-optimized data loader for efficient validation
        print("\nCreating GPU test dataloader...")
        
        # Create a temporary directory for intermediate data processing
        temp_dir = Path("./szakdolgozat-high-freq-btc-prediction/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename using timestamp to avoid conflicts
        temp_file = temp_dir / f"filtered_data_{int(time.time())}.parquet"
        
        # Save the combined filtered data to a temporary parquet file
        # This is needed for the GPU data loader to efficiently process the data
        print(f"Saving filtered data to temporary file: {temp_file}")
        self.combined_data.to_parquet(temp_file, index=False)
        
        # Create GPU-accelerated data loader for the test dataset
        # The GPU loader transfers data directly to GPU memory for faster inference
        _, self.test_loader = create_gpu_data_loaders(
            file_paths=[str(temp_file)],  # Path to the temporary data file
            valid_frac=1.0,  # Use all data as validation set (no training split)
            depth=depth,      # Number of LOB levels to consider
            window=window,    # Input time window size
            horizon=horizon,  # Prediction horizon
            batch_size=batch_size,  # Batch size for validation
            alpha=alpha,      # Threshold for price movement classification
            stride=1,        # Use stride=1 for validation to process all data points
            device=device     # Target device for data tensors
        )
        
        # Clean up the temporary file to save disk space
        try:
            os.remove(temp_file)
            print(f"Temporary file removed: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
        
        # Initialize loss function for validation
        self.criterion = nn.CrossEntropyLoss()  # Standard cross-entropy loss for 3-class classification
    
    def validate(self):
        """
        Perform model validation on the entire test dataset.
        
        Evaluates the pre-trained DeepLOB model on the loaded test data,
        calculates performance metrics including directional F1 score,
        macro F1 score, precision, recall, and confusion matrix.
        
        Returns:
            metrics: Dictionary containing comprehensive validation metrics
        """
        print("Starting validation phase...")
        val_start = time.time()
        self.model.eval()  # Set model to evaluation mode
        y_true, y_pred = [], []  # Lists to collect true labels and predictions
        test_loss = 0.0
        
        # Use torch.no_grad to disable gradient calculations for inference
        # Use mixed precision to improve performance on GPU
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(self.test_loader):
                # Print progress every 20 batches
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(self.test_loader)}")
                
                # Forward pass through the model
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                test_loss += loss.item()  # Accumulate batch loss
                batch_preds = logits.argmax(1)  # Get predicted class indices
                
                # Collect results for later analysis
                y_true.append(yb)
                y_pred.append(batch_preds)
        
        val_end = time.time()
        print(f"Validation completed in {val_end - val_start:.2f}s")
        
        # Concatenate all batch predictions and true labels into single tensors
        all_true = torch.cat(y_true)
        all_pred = torch.cat(y_pred)
        
        # Move tensors to CPU and convert to NumPy arrays for scikit-learn metrics
        cpu_true = all_true.cpu().numpy()
        cpu_pred = all_pred.cpu().numpy()
        
        # Initialize metrics dictionary with raw prediction results
        metrics = {'raw_predictions': cpu_pred}
        
        # Add timestamps and price data to metrics for visualization and analysis
        try:
            print("\nAdding timestamps and prices to metrics...")
            # Sample data based on the number of available predictions
            # This ensures alignment between predictions and time series data
            num_predictions = len(cpu_pred)
            metrics['timestamps'] = self.combined_data['received_time'].values[:num_predictions]
            metrics['normalized_prices'] = self.combined_data['ask_0_price'].values[:num_predictions]
            
            # Add raw (non-normalized) prices for reference and comparison
            if self.combined_raw_prices is not None:
                # Merge raw price data with normalized data based on timestamps
                # This ensures proper temporal alignment of different data sources
                merged_data = pd.merge(
                    pd.DataFrame({'received_time': metrics['timestamps']}),
                    self.combined_raw_prices,
                    on='received_time',
                    how='left'  # Left join to keep all prediction timestamps
                )
                metrics['raw_prices'] = merged_data['ask_0_price'].values
                print(f"Added raw prices to metrics. Shape: {len(metrics['raw_prices'])}")
            else:
                print("Warning: No raw prices available to add to metrics")
                # Provide zero array as placeholder when raw prices are unavailable
                metrics['raw_prices'] = np.zeros_like(metrics['normalized_prices'])
            
            # Report data shapes to verify alignment of different data arrays
            print(f"Data shapes: timestamps={len(metrics['timestamps'])}, "
                  f"normalized_prices={len(metrics['normalized_prices'])}, "
                  f"raw_prices={len(metrics['raw_prices'])}, "
                  f"predictions={len(cpu_pred)}")
            
        except Exception as e:
            # Robust error handling with full traceback for debugging
            print(f"Error adding timestamps and prices: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Calculate per-class F1 scores individually for detailed analysis
        class_f1 = f1_score(cpu_true, cpu_pred, average=None)
        
        # Extract F1 scores for each class: down (0), stable (1), and up (2)
        # These are important for analyzing model performance on each price movement direction
        f1_down, f1_stable, f1_up = class_f1
        
        # Calculate directional F1 score - average of up and down class F1 scores
        # This is a key financial metric focusing on direction prediction accuracy
        # rather than stable price prediction (which is less important for trading)
        directional_f1 = (f1_up + f1_down) / 2
        
        # Calculate standard macro F1 score (average of all three classes)
        # This provides a general classification performance metric for comparison
        macro_f1 = f1_score(cpu_true, cpu_pred, average='macro')
        
        # Average the validation loss across all batches
        test_loss = test_loss / len(self.test_loader)
        
        # Compute the confusion matrix for visualization and detailed analysis
        conf_matrix = confusion_matrix(cpu_true, cpu_pred)
        
        # Calculate precision and recall metrics per-class directly on GPU for efficiency
        # First create a confusion matrix on GPU
        conf_tensor = torch.zeros(3, 3, dtype=torch.int, device=device)
        for t in range(3):  # true class
            for p in range(3):  # predicted class
                conf_tensor[t, p] = torch.sum((all_true == t) & (all_pred == p))
        
        # Initialize tensors for precision and recall calculations
        precision = torch.zeros(3, device=device)
        recall = torch.zeros(3, device=device)
        
        # Calculate precision and recall for each class
        for i in range(3):
            # Precision: TP / (TP + FP) - correct predictions divided by total predictions for this class
            precision[i] = conf_tensor[i, i] / conf_tensor[:, i].sum() if conf_tensor[:, i].sum() > 0 else 0
            # Recall: TP / (TP + FN) - correct predictions divided by total actual instances of this class
            recall[i] = conf_tensor[i, i] / conf_tensor[i, :].sum() if conf_tensor[i, :].sum() > 0 else 0
        
        # Calculate F1 score directly from precision and recall
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Add small epsilon to avoid division by zero
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Analyze class distribution in validation dataset
        # This helps identify class imbalance issues
        class_counts = [torch.sum(all_true == i).item() for i in range(3)]
        class_distribution = [count / len(all_true) * 100 for count in class_counts]
        
        # Generate detailed classification report with precision, recall, and F1 for each class
        report = classification_report(
            cpu_true, 
            cpu_pred, 
            target_names=['Down', 'Stable', 'Up'],  # Human-readable class names
            digits=4,  # Show 4 decimal places for high precision
            output_dict=True  # Return as dictionary for easy access
        )
        
        # Collect all results into a comprehensive metrics dictionary for analysis and reporting
        metrics.update({
            'directional_f1': directional_f1,        # Key trading metric (avg of up/down F1 scores)
            'macro_f1': macro_f1,                  # Standard macro-average F1 across all classes
            'class_f1': class_f1,                  # Per-class F1 scores from scikit-learn
            'f1_per_class': f1_per_class.cpu().numpy(),  # GPU-calculated F1 scores per class
            'precision': precision.cpu().numpy(),   # Precision per class
            'recall': recall.cpu().numpy(),        # Recall per class
            'conf_matrix': conf_matrix,            # Confusion matrix for visualization
            'class_counts': class_counts,          # Raw count of instances per class
            'class_distribution': class_distribution,  # Percentage distribution of classes
            'report': report,                      # Detailed classification report dictionary
            'test_loss': test_loss,                # Average validation loss
            'all_true': cpu_true,                  # All ground truth labels
            'all_pred': cpu_pred                   # All model predictions
        })
        
        return metrics  # Return the complete metrics dictionary
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        Visualize the confusion matrix with detailed performance metrics.
        
        Creates a heatmap visualization of the confusion matrix showing the model's 
        classification performance across the three classes (down, stable, up).
        The plot includes directional and macro F1 scores in the title for reference.
        
        Args:
            metrics: Dictionary containing validation metrics returned by validate()
            save_path: Optional path to save the confusion matrix visualization as an image file
        """
        # Extract required metrics from the metrics dictionary
        conf_matrix = metrics['conf_matrix']
        directional_f1 = metrics['directional_f1']
        macro_f1 = metrics['macro_f1']
        
        # Create a figure with appropriate size for the confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Generate a heatmap visualization of the confusion matrix
        # with numeric annotations and color intensity representing frequency
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Down', 'Stable', 'Up'],
                   yticklabels=['Down', 'Stable', 'Up'])
        
        # Add a title with key performance metrics
        plt.title(f'Confusion Matrix (Directional F1: {directional_f1:.4f}, Macro F1: {macro_f1:.4f})')
        
        # Label axes for clarity
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save the figure if a save path is provided
        if save_path:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved confusion matrix to '{save_path}'")
        
        # Display the plot
        plt.show()
    
    def save_report(self, metrics, save_path):
        """
        Save a comprehensive validation report to a text file.
        
        Generates a detailed report of model performance including directional F1 score,
        macro F1 score, per-class precision/recall/F1, confusion matrix,
        and class distribution statistics. This report provides a complete
        overview of the model's prediction capabilities.
        
        Args:
            metrics: Dictionary containing validation metrics returned by validate()
            save_path: File path where the report should be saved
        """
        with open(save_path, 'w') as f:
            # Write basic model information and key performance metrics
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Directional F1: {metrics['directional_f1']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
            
            # Generate and include the scikit-learn classification report
            # This provides precision, recall, F1, and support for each class
            report_str = classification_report(
                metrics['all_true'],
                metrics['all_pred'],
                target_names=['Down', 'Stable', 'Up'],
                digits=4  # Use 4 decimal places for precision
            )
            f.write(f"Classification Report:\n{report_str}\n\n")
            
            # Include the confusion matrix for detailed error analysis
            f.write("Confusion Matrix:\n")
            f.write(str(metrics['conf_matrix']) + "\n\n")
            
            # Write detailed per-class metrics (precision, recall, F1)
            f.write("Per-class metrics:\n")
            class_names = ['Down', 'Stable', 'Up']
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall'][i]:.4f}\n")
                f.write(f"  F1: {metrics['f1_per_class'][i]:.4f}\n\n")
            
            # Include class distribution statistics to highlight dataset balance/imbalance
            f.write("Class distribution in test set:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name}: {metrics['class_counts'][i]} samples "
                        f"({metrics['class_distribution'][i]:.2f}%)\n")
        
        print(f"Saved detailed report to '{save_path}'")
        


def validate_model(start_date, end_date, model_path, 
                symbol='BTC-USDT', depth=10, window=100, horizon=100, 
                batch_size=64, alpha=0.002, stride=5, save_output=True,
                data_dir="./data_normalized"):
    """
    Validate a pre-trained DeepLOB model with specified parameters.
    
    This function serves as the main entry point for the validation pipeline.
    It loads data for the specified time period, initializes and runs the model validator,
    displays performance metrics, and optionally saves visualization and reports.
    
    Args:
        start_date: Start date for validation (dt.datetime object or string in 'YYYY-MM-DD' format)
        end_date: End date for validation (dt.datetime object or string in 'YYYY-MM-DD' format)
        model_path: Path to the pre-trained model (.pt file) to validate
        symbol: Trading symbol/pair to validate on (default: 'BTC-USDT')
        depth: Number of price levels in the limit order book (default: 10)
        window: Time window size for input features - number of timesteps used as input (default: 100)
        horizon: Prediction horizon - how many timesteps ahead to predict (default: 100)
        batch_size: Batch size for validation inference (default: 64)
        alpha: Threshold value for classifying price changes (default: 0.002)
                 0=down movement, 1=stable price, 2=up movement
        stride: Step size for sampling data points (default: 5)
        save_output: If True, confusion matrix and detailed report will be saved (default: True)
        data_dir: Directory containing normalized data files (default: "./data_normalized")
        
    Returns:
        metrics: Dictionary containing validation metrics and results
        validator: The ModelValidator instance for further analysis if needed
    """
    # Convert date strings to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Display validation configuration
    print(f"\n=== DeepLOB Model Validation ===")
    print(f"Time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Symbol: {symbol}")
    print(f"Model: {model_path}")
    
    # Search for normalized data files covering the requested time period
    t_start = time.time()
    file_infos = load_book_chunk(
        start_date,
        end_date,
        symbol,
        data_dir=data_dir
    )
    
    # Handle case where no data files are found
    if not file_infos:
        print("No normalized files found for the specified time period.")
        return None, None
    
    print(f"Found {len(file_infos)} files for processing")
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # Check if available data fully covers the requested time period
    total_start = min(info['start_date'] for info in file_infos)
    total_end = max(info['end_date'] for info in file_infos)
    
    # Warn user if available data doesn't fully cover the requested period
    if total_start > start_date or total_end < end_date:
        print(f"Warning: Available data only covers {total_start} to {total_end}")
        print("This is less than the requested time period.")
        response = input("Do you want to continue with the available data? (y/n): ")
        if response.lower() != 'y':
            print("Validation cancelled.")
            return None, None
    
    # Initialize the model validator with the configured parameters
    validator = ModelValidator(
        file_paths=file_infos,
        model_path=model_path,
        depth=depth,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        alpha=alpha,
        stride=stride
    )
    
    # Run the validation process
    metrics = validator.validate()
    
    # Display key validation results
    print("\n=== Validation Results ===")
    print(f"Directional F1 Score (average of down and up): {metrics['directional_f1']:.4f}")
    print(f"Global F1 Score (macro average): {metrics['macro_f1']:.4f}")
    print(f"Class F1 Scores - Down: {metrics['class_f1'][0]:.4f}, Stable: {metrics['class_f1'][1]:.4f}, Up: {metrics['class_f1'][2]:.4f}")
    
    # Save outputs if requested
    if save_output:
        # Create output directory structure if it doesn't exist
        output_dir = Path("./szakdolgozat-high-freq-btc-prediction/results/deeplob")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames with model name and timestamp
        model_name = os.path.basename(model_path).split('.')[0]
        date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save confusion matrix visualization
        conf_matrix_path = output_dir / f"validation_conf_matrix_{model_name}_{date_str}.png"
        validator.plot_confusion_matrix(metrics, save_path=str(conf_matrix_path))
        
        # Save detailed performance report
        report_path = output_dir / f"validation_report_{model_name}_{date_str}.txt"
        validator.save_report(metrics, str(report_path))
        
        # Save predictions to parquet file for further analysis/visualization
        print("\nChecking metrics for parquet file creation...")
        print(f"Available keys in metrics: {list(metrics.keys())}")
        
        # Check if all required fields are present to create the predictions file
        if all(key in metrics for key in ['timestamps', 'normalized_prices', 'raw_prices', 'raw_predictions']):
            print(f"Found required data for parquet file:")
            print(f"- Timestamps shape: {len(metrics['timestamps'])}")
            print(f"- Normalized prices shape: {len(metrics['normalized_prices'])}")
            print(f"- Raw prices shape: {len(metrics['raw_prices'])}")
            print(f"- Predictions shape: {len(metrics['raw_predictions'])}")
            
            # Create DataFrame with timestamps, predictions, and price data
            predictions_df = pd.DataFrame({
                'received_time': metrics['timestamps'],
                'prediction': metrics['raw_predictions'],
                'normalized_price': metrics['normalized_prices'],
                'raw_price': metrics['raw_prices']
            })
            
            # Save predictions to parquet file for efficient storage and later analysis
            predictions_path = output_dir / f"predictions_{model_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
            predictions_df.to_parquet(predictions_path, index=False)
            print(f"Saved predictions to '{predictions_path}'")
        else:
            # Handle missing data case
            print("Missing required data for parquet file:")
            missing_keys = [key for key in ['timestamps', 'normalized_prices', 'raw_prices', 'raw_predictions'] 
                          if key not in metrics]
            print(f"Missing keys: {missing_keys}")
    
    print(f"\nValidation completed successfully.")
    if save_output:
        print(f"Confusion matrix saved to '{conf_matrix_path}'")
        print(f"Detailed report saved to '{report_path}'")
        
    return metrics, validator

if __name__ == "__main__":
    # Example usage of the validation pipeline
    # This will run when the script is executed directly (not imported)
    validate_model(
        start_date="2025-03-05",  # Start date for validation period
        end_date="2025-03-10",    # End date for validation period
        model_path="./szakdolgozat-high-freq-btc-prediction/models/deeplob_single_parallel_f1_0.4369.pt"  # Path to pre-trained model
        # Uses default values for other parameters
    )
