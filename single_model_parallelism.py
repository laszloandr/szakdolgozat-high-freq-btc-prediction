"""
Single Model Parallelism - Training a single model with parallel processing across GPU partitions.

This module trains a DeepLOB (Deep Learning for Limit Order Book) model using data parallelism 
techniques to maximize GPU utilization. The implementation divides batches into micro-batches 
that can be processed in parallel, improving training efficiency without increasing memory requirements.
"""
import os, datetime as dt, time
from pathlib import Path

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PyTorch and GPU acceleration libraries
import torch
from torch import nn
from torch.cuda import amp  # Automatic mixed precision
import torch.cuda.amp as amp
import torch.nn.functional as F

# Machine learning evaluation metrics
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Import the DeepLOB model and data loading functions from our optimized implementation
from deeplob_optimized import (
    DeepLOB,  # The neural network architecture for Limit Order Book data
    load_book_chunk  # Function to load preprocessed LOB data chunks
)

# Import GPU-optimized dataset handling utilities
from gpu_loaders import create_gpu_data_loaders, process_file_infos

# PyTorch and CUDA configuration
# Set up the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Performance optimization settings
torch.backends.cudnn.benchmark = True  # Optimize CUDNN operations for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32 for faster computation
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 in cuDNN operations

class SingleModelParallelTrainer:
    """
    Single Model Trainer that utilizes data parallelism techniques to maximize GPU utilization.
    
    This class implements a training framework for a single DeepLOB model using micro-batch
    parallelism. Instead of processing one large batch sequentially, it splits each batch
    into multiple smaller micro-batches that can be processed in parallel on the GPU,
    leading to better hardware utilization and faster training times.
    """
    def __init__(self, 
                 file_paths,
                 depth=10,
                 window=100,
                 horizon=100,
                 batch_size=64,
                 alpha=0.002,
                 stride=5,
                 epochs=40,
                 lr=1e-3,
                 patience=5,
                 split_batches=3):  # Number of parallel micro-batches to split each batch into
        """
        Initializes the parallel data processing trainer with specified parameters.
        
        Args:
            file_paths: List of data files to process (normalized parquet files)
            depth: Number of price levels in the limit order book (LOB depth)
            window: Size of the time window for input features (number of time steps)
            horizon: Prediction horizon for future price movements (number of time steps ahead)
            batch_size: Base batch size for training
            alpha: Threshold value for classifying price changes (0=down, 1=stable, 2=up)
            stride: Step size for sampling data points (controls overlap between samples)
            epochs: Maximum number of training epochs
            lr: Learning rate for the optimizer
            patience: Number of epochs to wait without improvement before early stopping
            split_batches: Number of parallel micro-batches to divide each batch into for
                           more efficient GPU processing
        """
        # Store all configuration parameters as instance variables
        self.file_paths = file_paths
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.alpha = alpha
        self.stride = stride
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.split_batches = split_batches
        
        # Initialize the DeepLOB model
        print("Initializing model...")
        # Create model and move it to GPU with channels-last memory format for better performance
        self.model = DeepLOB(depth=depth).to(
            device, 
            memory_format=torch.channels_last  # This improves performance for convolutional operations
        )
        
        # Create GPU-optimized data loaders
        print("Creating GPU dataloaders...")
        
        # Load the entire dataset directly into GPU memory for faster access during training
        # This is a key optimization that eliminates CPU-GPU transfer bottlenecks
        self.train_loader, self.val_loader = create_gpu_data_loaders(
            file_paths=file_paths,
            valid_frac=0.1,  # 10% of data used for validation
            depth=depth,
            window=window,
            horizon=horizon,
            # Use larger batches that will be split into micro-batches during training
            batch_size=batch_size * split_batches,
            alpha=alpha,
            stride=stride,
            device=device  # Target device for data loading (GPU)
        )
        
        # Initialize optimizer with AdamW (Adam with weight decay fix)
        # AdamW provides better generalization compared to standard Adam
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,  # Learning rate
            betas=(0.9, 0.999),  # Exponential decay rates for moment estimates
            eps=1e-8,  # Small constant for numerical stability
            weight_decay=1e-4,  # L2 regularization strength
            fused=True  # Use fused implementation for better performance
        )
        
        # Initialize mixed precision scaling for automated mixed precision training
        # This allows the use of faster FP16 operations where possible while maintaining model accuracy
        self.scaler = amp.GradScaler(
            growth_factor=2.0,  # Factor by which to increase scale when no overflow occurs
            backoff_factor=0.5,  # Factor by which to decrease scale when overflow occurs
            growth_interval=2000,  # Number of consecutive unskipped steps before increasing scale
            enabled=True  # Enable mixed precision training
        )
        
        # Loss function: standard cross-entropy loss for multi-class classification
        # Our target classes are: 0=down, 1=stable, 2=up
        self.criterion = nn.CrossEntropyLoss()
    
    def _parallel_forward(self, x, y):
        """
        Performs parallel forward pass across multiple micro-batches to maximize GPU utilization.
        
        This method splits a large batch into smaller micro-batches that can be processed
        in parallel on the GPU. This improves computational efficiency by ensuring better
        occupancy of GPU resources. The losses from each micro-batch are normalized and
        then aggregated.
        
        Args:
            x: Input batch tensor already loaded on GPU (shape: [batch_size, channels, time_steps, features])
            y: Target labels tensor already loaded on GPU (shape: [batch_size])
            
        Returns:
            tuple: (total_loss, all_logits) where total_loss is the combined loss across all
                   micro-batches and all_logits contains the model's output predictions
        """
        # Calculate the size of each micro-batch by dividing the full batch
        micro_batch_size = x.size(0) // self.split_batches
        
        # Initialize lists to store losses and predictions from each micro-batch
        losses = []
        all_logits = []
        
        # Enable automatic mixed precision for faster computation
        # This allows operations to use FP16 where possible while maintaining accuracy
        with torch.amp.autocast(device_type='cuda'):
            # Process each micro-batch in sequence (though operations within each are parallel)
            for i in range(self.split_batches):
                # Calculate the start and end indices for this micro-batch
                start_idx = i * micro_batch_size
                # For the last batch, include any remaining elements (handles uneven division)
                end_idx = (i + 1) * micro_batch_size if i < self.split_batches - 1 else x.size(0)
                
                # Extract this micro-batch's data and ensure memory is contiguous for efficient processing
                micro_x = x[start_idx:end_idx].contiguous()
                micro_y = y[start_idx:end_idx]
                
                # Forward pass: get model predictions for this micro-batch
                logits = self.model(micro_x)  # Shape: [micro_batch_size, num_classes]
                # Calculate loss and normalize by the number of micro-batches
                # This ensures the gradient magnitudes are comparable regardless of split_batches value
                loss = self.criterion(logits, micro_y) / self.split_batches
                
                # Store results for later aggregation
                losses.append(loss)
                all_logits.append(logits)
        
        # Sum the normalized losses from all micro-batches
        # Since each loss was divided by split_batches, this maintains the correct scale
        total_loss = sum(losses)
        
        # Concatenate predictions from all micro-batches to recreate the full batch output
        # This ensures the output logits match the original input batch size
        all_logits = torch.cat(all_logits, dim=0)
        
        return total_loss, all_logits
    
    def train_one_epoch(self, epoch):
        """
        Trains the model for one complete epoch using parallel data processing techniques.
        
        This method iterates through all batches in the training data loader, performing
        forward and backward passes with mixed precision and gradient accumulation to optimize
        memory usage and computational efficiency. Performance metrics are tracked and
        periodically reported.
        
        Args:
            epoch: Current epoch number (1-based indexing)
            
        Returns:
            float: Average loss value across the entire epoch
        """
        # Set model to training mode (enables dropout, batch normalization updates, etc.)
        self.model.train()
        running_loss = 0.0
        
        # Initialize timing metrics for performance monitoring
        batch_times = []    # Total time per batch
        forward_times = []  # Time for forward pass
        backward_times = [] # Time for backward pass
        
        # Gradient accumulation settings
        # This allows us to effectively increase batch size without increased memory usage
        accum_steps = 4  # Update weights after every 4 mini-batches
        
        print(f"Training: {len(self.train_loader)} batches with split_batches={self.split_batches}")
        
        # Iterate through all batches in the training data loader
        for step, (xb, yb) in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Forward pass using the parallel processing implementation
            forward_start = time.time()
            loss, _ = self._parallel_forward(xb, yb)  # Get loss and ignore logits
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Backward pass with automatic mixed precision scaling
            backward_start = time.time()
            self.scaler.scale(loss).backward()  # Scale loss to prevent underflow in fp16
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            
            # Gradient accumulation - only update weights every accum_steps iterations
            if (step + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer)  # Update weights with accumulated gradients
                self.scaler.update()  # Update the scale factor for next iteration
                self.optimizer.zero_grad(set_to_none=True)  # Clear gradients (memory-efficient way)
            
            # Accumulate loss value (multiplied by accumulation steps to normalize for reporting)
            running_loss += loss.item() * accum_steps
            
            # Measure batch processing time
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Display status update every 50 batches
            if (step + 1) % 50 == 0:
                # Calculate average batch processing time from recent batches
                avg_batch = sum(batch_times[-50:]) / min(50, len(batch_times[-50:]))
                
                # Estimate total epoch time (elapsed time + remaining batches * avg batch time)
                remaining_batches = len(self.train_loader) - (step + 1)
                estimated_remaining_time = remaining_batches * avg_batch
                elapsed_time = time.time() - batch_start + sum(batch_times[:-1])
                estimated_total_time = elapsed_time + estimated_remaining_time
                
                # Print progress information including loss and estimated completion time
                print(f"Epoch {epoch} - Batch {step+1}/{len(self.train_loader)} - "
                      f"Loss: {running_loss/(step+1):.4f} - "
                      f"Estimated epoch time: {estimated_total_time/60:.2f} minutes")
                
                # Monitor GPU memory usage to detect potential memory issues
                print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
                      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # Return the average loss for the entire epoch
        # Divide by actual number of samples rather than batches to handle uneven batch sizes
        return running_loss / len(self.train_loader.sampler)
    
    def validate(self):
        """
        Evaluates the model on validation data using parallel processing and calculates detailed metrics.
        
        This method runs the model in evaluation mode on the validation dataset, calculating
        comprehensive performance metrics with a focus on directional prediction accuracy.
        It uses the same parallel processing approach as training for consistent evaluation.
        
        The method calculates class-specific metrics and specially tracks the 'directional F1 score',
        which is the average of F1 scores for the 'up' and 'down' classes, ignoring the 'stable' class.
        This is a domain-specific metric that's more relevant for financial prediction tasks where
        correctly predicting price direction (up/down) is more important than stability.
        
        Returns:
            tuple: (
                directional_f1: F1 score focused on directional accuracy (average of 'up'/'down' classes),
                val_loss: Average validation loss value,
                metrics: Dictionary containing detailed performance metrics and raw predictions
            )
        """
        print("Starting validation phase...")
        val_start = time.time()
        # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
        self.model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        
        # Disable gradient calculation and use mixed precision for validation
        # This saves memory and speeds up inference
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(self.val_loader):
                # Progress reporting
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(self.val_loader)}")
                    
                # Use the same parallel forward implementation as in training
                loss, logits = self._parallel_forward(xb, yb)
                val_loss += loss.item()  # Accumulate batch loss
                # Convert logits to class predictions by taking argmax
                batch_preds = logits.argmax(1)  # Shape: [batch_size]
                
                # Collect predictions and true labels for later evaluation
                y_true.append(yb)
                y_pred.append(batch_preds)
        
        # Report validation time
        val_end = time.time()
        print(f"Validation completed in {val_end - val_start:.2f}s")
        
        # Concatenate all batch results into single tensors
        all_true = torch.cat(y_true)  # All ground truth labels
        all_pred = torch.cat(y_pred)  # All model predictions
        
        # Transfer to CPU for scikit-learn metric calculations
        # GPU tensors aren't compatible with sklearn functions
        cpu_true = all_true.cpu().numpy()
        cpu_pred = all_pred.cpu().numpy()
        
        # Calculate F1 scores for each class (returns array with score for each class)
        # Class mapping: 0=down, 1=stable, 2=up 
        class_f1 = f1_score(cpu_true, cpu_pred, average=None)
        
        # Unpack F1 scores for each class for easier reference
        f1_down, f1_stable, f1_up = class_f1
        
        # Calculate directional F1 score - the average of up and down class F1 scores
        # This is our primary metric because it focuses on directional accuracy
        # which is more important than correctly identifying stable periods
        directional_f1 = (f1_up + f1_down) / 2
        
        # Also calculate standard macro F1 for comparison and diagnostics
        # This includes the 'stable' class in the average
        macro_f1 = f1_score(cpu_true, cpu_pred, average='macro')
        
        # Calculate average validation loss
        val_loss = val_loss / len(self.val_loader)
        
        # Generate confusion matrix for detailed error analysis
        conf_matrix = confusion_matrix(cpu_true, cpu_pred)
        
        # Calculate precision and recall on GPU for efficiency
        # First build confusion matrix on GPU
        conf_tensor = torch.zeros(3, 3, dtype=torch.int, device=device)
        for t in range(3):  # True class
            for p in range(3):  # Predicted class
                # Count matches between true and predicted classes
                conf_tensor[t, p] = torch.sum((all_true == t) & (all_pred == p))
        
        # Initialize precision and recall tensors
        precision = torch.zeros(3, device=device)
        recall = torch.zeros(3, device=device)
        
        # Calculate precision and recall for each class
        for i in range(3):
            # Precision: True Positives / (True Positives + False Positives)
            # Column sum = all predictions of class i
            precision[i] = conf_tensor[i, i] / conf_tensor[:, i].sum() if conf_tensor[:, i].sum() > 0 else 0
            
            # Recall: True Positives / (True Positives + False Negatives)
            # Row sum = all actual instances of class i
            recall[i] = conf_tensor[i, i] / conf_tensor[i, :].sum() if conf_tensor[i, :].sum() > 0 else 0
        
        # Calculate F1 score from precision and recall: 2 * (P * R) / (P + R)
        # Add small epsilon to prevent division by zero
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Calculate class distribution in validation set
        class_counts = [torch.sum(all_true == i).item() for i in range(3)]
        class_distribution = [count / len(all_true) * 100 for count in class_counts]
        
        # Generate detailed classification report
        report = classification_report(
            cpu_true, 
            cpu_pred, 
            target_names=['Down', 'Stable', 'Up'],  # Human-readable class names 
            digits=4,  # Display 4 decimal places for precision
            output_dict=True  # Return as dictionary for easier processing
        )
        
        # Collect all metrics into a single dictionary for easy access
        metrics = {
            'directional_f1': directional_f1,  # Primary metric (up/down average)
            'macro_f1': macro_f1,  # Secondary metric (all classes)
            'class_f1': class_f1,  # Individual class F1 scores from sklearn
            'f1_per_class': f1_per_class.cpu().numpy(),  # Our calculated per-class F1 scores
            'precision': precision.cpu().numpy(),  # Per-class precision
            'recall': recall.cpu().numpy(),  # Per-class recall
            'conf_matrix': conf_matrix,  # Confusion matrix
            'class_counts': class_counts,  # Number of samples in each class
            'class_distribution': class_distribution,  # Percentage distribution of classes
            'report': report,  # Full classification report as dictionary
            'val_loss': val_loss,  # Validation loss
            'all_true': cpu_true,  # All ground truth labels
            'all_pred': cpu_pred  # All predictions
        }
        
        return directional_f1, val_loss, metrics
    
    def train(self):
        """
        Main method to run the complete model training process with early stopping.
        
        This method orchestrates the entire training process, including:
        - Running training epochs
        - Evaluating model performance on validation data
        - Tracking best model based on directional F1 score
        - Implementing early stopping for training efficiency
        - Saving model checkpoints and performance visualizations
        - Generating detailed performance reports
        
        The method focuses on the directional F1 score (average of up/down F1 scores)
        as the primary metric for model selection and early stopping decisions.
        
        Returns:
            None: The method doesn't return the model as it's kept within the trainer object
        """
        print(f"\n=== Starting Training with Parallel Data Processing ===\n")
        start_time = time.time()
        
        # Initialize tracking variables for training monitoring
        best_f1 = 0.0  # Best directional F1 score achieved
        wait = 0       # Counter for early stopping patience
        total_train_time = 0  # Total training time across all epochs
        
        # Main training loop - iterate through epochs
        for ep in range(1, self.epochs + 1):
            epoch_start = time.time()
            print(f"\n--- Epoch {ep}/{self.epochs} ---")
            
            # Run one complete training epoch
            loss = self.train_one_epoch(ep)
            
            # Evaluate model on validation data
            directional_f1, val_loss, metrics = self.validate()
            
            # Calculate epoch statistics
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            
            # Extract specific metrics for reporting
            macro_f1 = metrics['macro_f1']  # Average F1 across all classes
            class_f1 = metrics['class_f1']  # F1 scores for each individual class
            conf_matrix = metrics['conf_matrix']  # Confusion matrix
            f1_down, f1_stable, f1_up = class_f1  # Unpack class-specific F1 scores
            
            # Display epoch performance metrics
            print(f"Epoch {ep} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"F1 Scores - Down: {f1_down:.4f}, Stable: {f1_stable:.4f}, Up: {f1_up:.4f}")
            print(f"Directional F1: {directional_f1:.4f}, Macro F1: {macro_f1:.4f}")
            
            # Periodically display detailed classification report (first epoch and every 10th epoch)
            if ep % 10 == 0 or ep == 1:
                report_str = classification_report(
                    metrics['all_true'],
                    metrics['all_pred'],
                    target_names=['Down', 'Stable', 'Up'],
                    digits=4
                )
                print(f"Classification Report:\n{report_str}")
            
            # Check if this is the best model so far
            # Using a small epsilon (1e-4) to avoid saving for negligible improvements
            if directional_f1 > best_f1 + 1e-4:  # Using directional F1 as our primary metric
                best_f1 = directional_f1  # Update best score
                wait = 0  # Reset early stopping counter
                
                # Create model directory if it doesn't exist
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                
                # Define checkpoint filename with F1 score for easy identification
                checkpoint_path = Path(f"models/deeplob_single_parallel_f1_{best_f1:.4f}.pt")
                
                # Prepare model state dictionary with all relevant information
                model_state = {
                    'state_dict': self.model.state_dict(),  # Model weights
                    'optimizer': self.optimizer.state_dict(),  # Optimizer state
                    'epoch': ep,  # Current epoch number
                    'f1': best_f1,  # Best F1 score
                    'directional_f1': directional_f1,  # Directional F1 score
                    'macro_f1': macro_f1  # Macro F1 score
                }
                
                # Save model checkpoint
                torch.save(model_state, checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
                
                # Generate and save confusion matrix visualization
                plt.figure(figsize=(10, 8))
                # Create heatmap with annotations showing exact counts
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                           xticklabels=['Down', 'Stable', 'Up'],
                           yticklabels=['Down', 'Stable', 'Up'])
                plt.title(f'Best Confusion Matrix (F1: {directional_f1:.4f})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig('best_confusion_matrix_single_parallel.png')
                plt.close()
                print(f"Saved confusion matrix to 'best_confusion_matrix_single_parallel.png'")
                
                # Save detailed metrics to text file for further analysis
                report_path = f"models/classification_report_single_parallel.txt"
                with open(report_path, 'w') as f:
                    # Write basic information
                    f.write(f"Epoch: {ep}\n")
                    f.write(f"Directional F1: {directional_f1:.4f}\n")
                    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
                    
                    # Write classification report
                    f.write(f"Classification Report:\n{report_str}\n\n")
                    
                    # Write confusion matrix
                    f.write("Confusion Matrix:\n")
                    f.write(str(conf_matrix) + "\n\n")
                    
                    # Write per-class metrics for detailed analysis
                    f.write("Per-class metrics:\n")
                    class_names = ['Down', 'Stable', 'Up']
                    for i, class_name in enumerate(class_names):
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
                        f.write(f"  Recall: {metrics['recall'][i]:.4f}\n")
                        f.write(f"  F1: {metrics['f1_per_class'][i]:.4f}\n\n")
                    
                    # Write class distribution information
                    f.write("Class distribution in validation set:\n")
                    for i, class_name in enumerate(class_names):
                        f.write(f"  {class_name}: {metrics['class_counts'][i]} samples "
                                f"({metrics['class_distribution'][i]:.2f}%)\n")
                
                print(f"Saved detailed metrics to '{report_path}'")
                
            else:
                # If no improvement, increment wait counter for early stopping
                wait += 1
                print(f"No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
                
                # Check if early stopping criteria is met
                if wait >= self.patience:
                    print(f"Early stopping after {ep} epochs")
                    break
        
        # Report final training statistics
        print(f"\nTraining completed in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # We don't return the model as it's accessible within the trainer object
        return

def main():
    """
    Main entry point for the DeepLOB model training with parallel data processing.
    
    This function orchestrates the entire training workflow:
    1. Locates and loads normalized data files for the specified time period and trading pair
    2. Initializes the parallel trainer with optimized configuration parameters
    3. Executes the training process
    4. Returns the trained model trainer object for potential further use
    
    The function includes performance timing for each major step to help track and optimize
    the training pipeline efficiency.
    
    Returns:
        SingleModelParallelTrainer: The trainer object containing the trained model and associated state
    """
    # Display header to indicate start of training process
    print("\n=== DeepLOB Training - Single Model Parallelism ===\n")
    
    # Step 1: Locate normalized data files within the specified date range
    t_start = time.time()
    # Load book chunk files for BTC-USDT pair in the specified date range
    # These are normalized parquet files containing processed limit order book data
    file_infos = load_book_chunk(
            dt.datetime(2024, 11, 1),   # Start date for training data
            dt.datetime(2025, 2, 28),   # End date for training data
            "BTC-USDT")                  # Trading pair to analyze
    
    # Verify that required data files exist
    if not file_infos:
        print("No normalized files found. Please run normalize_data.py first.")
        return
    
    # Convert file information to absolute file paths for loading
    file_paths = process_file_infos(file_infos)
    print(f"Found {len(file_paths)} files for processing")
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # Step 2: Initialize the parallel trainer with optimized configuration
    print("\nInitializing single model parallel trainer...")
    t_start = time.time()
    
    # Create trainer instance with a single model but parallel data processing
    # This approach maximizes GPU utilization while keeping the model architecture simple
    trainer = SingleModelParallelTrainer(
        file_paths=file_paths,  # List of normalized parquet files to process
        depth=10,               # Number of price levels in the limit order book
        window=100,             # Input time window size (number of time steps)
        horizon=100,            # Prediction horizon (number of time steps ahead)
        batch_size=64,          # Base batch size (will be multiplied by split_batches)
        alpha=0.002,            # Threshold for price movement classification
        stride=5,               # Step size for sampling data points
        epochs=40,              # Maximum number of training epochs
        lr=1e-3,                # Learning rate for optimizer
        patience=5,             # Early stopping patience (epochs without improvement)
        split_batches=4         # Number of parallel micro-batches for data processing
    )
    
    print(f"Trainer initialization completed in {time.time()-t_start:.2f}s")
    
    # Step 3: Execute the training process with the parallel data processing
    trainer.train()
    
    print("\n=== Training Complete! ===")
    
    # Return the trainer object which contains the trained model
    # This allows for further use if this function is called programmatically
    return trainer
