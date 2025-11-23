import torch
import psutil
import GPUtil
import subprocess
import sys

def setup_device():
    """
    Setup and return the best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def to_device(data, device):
    """
    Move data to specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif hasattr(data, 'to'):
        return data.to(device)
    else:
        return data

def clear_gpu_memory():
    """
    Clear GPU memory cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cleared")

def print_gpu_memory():
    """
    Print GPU memory usage
    """
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total_memory = gpu_props.total_memory / 1024**3
        
        print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total_memory:.2f}GB")
        
        # Print per-process GPU memory if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[gpu_id]
                print(f"📊 GPU Utilization: {gpu.load*100:.1f}%, Memory Used: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
        except:
            pass

def check_cuda_installation():
    """
    Check CUDA installation and compatibility
    """
    print("🔍 Checking CUDA Installation...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA Available: {cuda_available}")
    
    if cuda_available:
        # CUDA device info
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"🔧 Number of GPUs: {device_count}")
        print(f"🎯 Current GPU: {device_name}")
        print(f"📋 CUDA Version: {torch.version.cuda}")
        
        # Check CUDA capabilities
        major = torch.cuda.get_device_capability(current_device)[0]
        print(f"⚡ Compute Capability: {major}.x")
        
        # Memory info
        print_gpu_memory()
        
    else:
        print("❌ CUDA not available. Please check your installation.")
        
        # Check if CUDA drivers are installed
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("⚠️  NVIDIA drivers are installed but PyTorch CUDA is not working")
                print("💡 Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("❌ NVIDIA drivers not found")
        except:
            print("❌ NVIDIA drivers not found")
    
    return cuda_available

def optimize_gpu_usage():
    """
    Optimize GPU usage settings
    """
    if torch.cuda.is_available():
        # Enable memory efficient algorithms
        torch.backends.cudnn.benchmark = True
        
        # Set memory growth if available
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        except:
            pass
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print("⚡ GPU optimization completed")
    else:
        print("💻 Running on CPU - no GPU optimization needed")

def get_batch_size_recommendation(model, input_shape, max_memory_fraction=0.8):
    """
    Recommend optimal batch size based on available GPU memory
    """
    if not torch.cuda.is_available():
        return 32  # Default for CPU
    
    try:
        # Get available GPU memory
        gpu_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        available_memory = total_memory * max_memory_fraction
        
        # Estimate memory per sample
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(*input_shape).cuda()
            
            # Forward pass to estimate memory
            model = model.cuda()
            output = model(dummy_input)
            
            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_per_sample = memory_allocated / input_shape[0]
            
            # Calculate recommended batch size
            recommended_batch_size = int(available_memory / memory_per_sample)
            
            # Clear memory
            del dummy_input, output
            torch.cuda.empty_cache()
            
            # Apply safety margin
            recommended_batch_size = max(1, recommended_batch_size // 2)
            
            print(f"🎯 Recommended batch size: {recommended_batch_size}")
            return recommended_batch_size
            
    except Exception as e:
        print(f"⚠️ Batch size recommendation failed: {e}")
        return 16  # Fallback batch size

def monitor_system_resources():
    """
    Monitor system resources (CPU, RAM, GPU)
    """
    # CPU and RAM usage
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_used_gb = ram.used / 1024**3
    ram_total_gb = ram.total / 1024**3
    
    print(f"💻 CPU Usage: {cpu_percent:.1f}%")
    print(f"🧠 RAM Usage: {ram_percent:.1f}% ({ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB)")
    
    # GPU usage if available
    if torch.cuda.is_available():
        print_gpu_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    disk_used_gb = disk.used / 1024**3
    disk_total_gb = disk.total / 1024**3
    
    print(f"💾 Disk Usage: {disk_percent:.1f}% ({disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB)")

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import numpy as np
    np.random.seed(seed)
    
    import random
    random.seed(seed)
    
    print(f"🎲 Random seeds set to: {seed}")

def get_device_info():
    """
    Get comprehensive device information
    """
    info = {}
    
    # CPU info
    info['cpu_count'] = psutil.cpu_count()
    info['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
    info['ram_total_gb'] = psutil.virtual_memory().total / 1024**3
    
    # GPU info
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        info['gpu_name'] = torch.cuda.get_device_name(gpu_id)
        info['gpu_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
    else:
        info['gpu_available'] = False
    
    # Python and PyTorch info
    info['python_version'] = sys.version.split()[0]
    info['pytorch_version'] = torch.__version__
    
    return info

def print_device_info():
    """
    Print comprehensive device information
    """
    info = get_device_info()
    
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 40)
    print(f"💻 CPU: {info['cpu_count']} cores @ {info['cpu_freq']} MHz")
    print(f"🧠 RAM: {info['ram_total_gb']:.1f} GB")
    
    if 'gpu_name' in info:
        print(f"🚀 GPU: {info['gpu_name']}")
        print(f"🔧 CUDA: {info['cuda_version']}")
        print(f"💾 GPU Memory: {info['gpu_memory_gb']:.1f} GB")
        print(f"🎯 Number of GPUs: {info['gpu_count']}")
    else:
        print("❌ No GPU available")
    
    print(f"🐍 Python: {info['python_version']}")
    print(f"🔥 PyTorch: {info['pytorch_version']}")
    print("=" * 40)