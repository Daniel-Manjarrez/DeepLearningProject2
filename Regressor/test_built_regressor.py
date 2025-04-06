import torch
import cv2
import os
import numpy as np
from regressor import CNNRegressor

def predict(image_path, model, device):
    # Prepare the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match input size
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = L.astype(np.float32) / 100.0  # Normalize to [0, 1]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension

    # Make prediction
    with torch.no_grad():
        model.eval()
        prediction = model(L_tensor.to(device))
    
    return prediction.cpu().numpy(), a.mean() - 128.0, b.mean() - 128.0  # Predicted values and actual mean a* and b*

if __name__ == "__main__":
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRegressor().to(device)
    model.load_state_dict(torch.load("best_cnn_regressor.pth"))
    
    # Get the image directory
    base_dir = os.getcwd()
    image_dir = os.path.join(base_dir, "augmented")
    
    # Get the first 10 images from the directory
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                   if fname.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
    
    # Loop through the first 10 images
    for image_path in image_paths:
        prediction, actual_a, actual_b = predict(image_path, model, device)
        
        # Print comparison between prediction and actual values
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted mean chrominance: a* = {prediction[0][0]:.4f}, b* = {prediction[0][1]:.4f}")
        print(f"Actual mean chrominance: a* = {actual_a:.4f}, b* = {actual_b:.4f}")
        print("-" * 50)
