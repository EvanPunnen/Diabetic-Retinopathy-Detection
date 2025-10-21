import torch
import sys
import platform
import subprocess

def check_cuda():
    print("\n=== CUDA Configuration Check ===\n")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available", file=sys.stderr)
        print("Please check your installation and drivers.\n", file=sys.stderr)
        return
    
    # CUDA version information
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # CUDNN version
    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN version: {cudnn_version}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    except:
        print("cuDNN version: Not available")
    
    # GPU device information
    device_count = torch.cuda.device_count()
    print(f"GPU device count: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\nCurrent CUDA device: {current_device}")
    
    # Test CUDA tensor operations
    print("\nTesting CUDA tensor operations...")
    try:
        # Create a CUDA tensor
        x = torch.ones(10, device='cuda')
        y = x * 2
        # Force synchronization
        torch.cuda.synchronize()
        print("✓ CUDA tensor operations are working correctly")
    except Exception as e:
        print(f"✗ CUDA tensor operations failed: {e}")
    
    print("\n=== CUDA Configuration Check Complete ===")

if __name__ == "__main__":
    check_cuda()