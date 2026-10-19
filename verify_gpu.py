import sys
import torch

def verify_gpu():
    print("=" * 40)
    print("GPU VERIFICATION REPORT")
    print("=" * 40)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"  VRAM: {total_mem_gb:.2f} GB")
        
        # Perform a small CUDA tensor test
        print("-" * 40)
        print("Running CUDA tensor test...")
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            # Sync to ensure it ran on GPU
            torch.cuda.synchronize()
            print(f"CUDA tensor multiplication successful! Shape: {z.shape}")
            print(f"Device of result: {z.device}")
            print("CUDA test result: PASSED")
        except Exception as e:
            print(f"CUDA test result: FAILED with error: {e}")
    else:
        print("CUDA is NOT available. PyTorch cannot use the GPU.")
        print("CUDA test result: FAILED")
    print("=" * 40)

if __name__ == "__main__":
    verify_gpu()
