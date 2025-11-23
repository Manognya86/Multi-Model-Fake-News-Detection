import torch
import subprocess
import platform

def check_gpu():
    """Check GPU availability and details"""
    print("=" * 60)
    print("GPU DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check PyTorch CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("\nPyTorch cannot detect CUDA. Possible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers are not installed")
        print("3. CUDA toolkit is not installed")
        print("4. GPU is not compatible with CUDA")
    
    # Check system-level GPU information
    print("\n" + "=" * 30)
    print("SYSTEM GPU INFORMATION")
    print("=" * 30)
    
    try:
        if platform.system() == "Windows":
            # Try to run nvidia-smi on Windows
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("NVIDIA GPU Information:")
                print(result.stdout)
            else:
                print("nvidia-smi not available or failed")
                
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Mac GPU Information:")
                # Extract GPU info from system_profiler output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chipset Model' in line or 'VRAM' in line:
                        print(line.strip())
        else:  # Linux
            result = subprocess.run(['lspci', '|', 'grep', '-i', 'vga'], 
                                  shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("GPU Information:")
                print(result.stdout)
    except Exception as e:
        print(f"Error getting system GPU info: {e}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not cuda_available:
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
        print("2. Update NVIDIA drivers")
        print("3. Install CUDA Toolkit from NVIDIA")
    else:
        print("✅ Your GPU is properly configured for PyTorch!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

if __name__ == '__main__':
    check_gpu()