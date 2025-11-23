import torch
import gc
import os

def clear_gpu_memory():
    """Clear GPU memory completely"""
    print("Clearing GPU memory...")
    
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Try to release all unused memory
        torch.cuda.synchronize()
        
        # Show before and after memory usage
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        
        print(f"Memory freed: {allocated_before - allocated_after:.2f} GB")
        print(f"Reserved memory freed: {reserved_before - reserved_after:.2f} GB")
        print(f"Current allocated: {allocated_after:.2f} GB")
        print(f"Current reserved: {reserved_after:.2f} GB")
    else:
        print("No GPU detected")
    
    print("GPU memory cleared successfully")

if __name__ == "__main__":
    clear_gpu_memory()