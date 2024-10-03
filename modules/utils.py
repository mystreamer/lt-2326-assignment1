import torch

def detect_platform(cuda_num):
    if torch.cuda.is_available():
        print("cuda is available")
        return f'cuda:{cuda_num}'
    elif torch.backends.mps.is_available():
        print("mps is available")
        return 'mps'
    else:
        print("cpu is available")
        return 'cpu'
