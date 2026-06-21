import torch
from model import ResNetMini

def load_classifier(model_path="checkpoints/classifier_mnist_resnet.pth", device=None):
    """
    Initializes the ResNetMini architecture and loads saved weights.
    
    Args:
        model_path (str): Path to the .pth file.
        device (str): 'cuda' or 'cpu'. Auto-detects if None.
        
    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Initialize the architecture
    model = ResNetMini()

    # 2. Load the state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Did you run train_classifier.py first?")
        return None

    # 3. Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Quick test to verify loading works
    my_model = load_classifier()
    if my_model:
        print("Model is ready for inference!")
