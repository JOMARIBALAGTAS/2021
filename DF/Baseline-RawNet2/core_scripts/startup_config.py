import torch
import numpy as np
import random
import os


def set_random_seed(seed, args=None):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
        args (object, optional): Arguments object that may contain 
            cudnn-deterministic and cudnn-benchmark settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Check if args has cudnn settings
        if args is not None:
            # Handle possible attribute errors
            if hasattr(args, 'cudnn_deterministic_toggle'):
                torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
            if hasattr(args, 'cudnn_benchmark_toggle'):
                torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle
        else:
            # Default settings for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
