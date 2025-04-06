import torch
import cv2
import os
import numpy as np
from regressor import CNNRegressor

def load_trained_model():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRegressor().to(device)
    model.load_state_dict(torch.load("best_cnn_regressor.pth"))
    
    return model, device

def predict(image_path):
    # Load model and device
    model, device = load_trained_model()
    
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

    
    # Get the image filename from the user
    image_filename = input("Enter the image filename (e.g., image1.jpg): ")
    base_dir = os.getcwd()
    image_dir = os.path.join(base_dir, "augmented")
    image_path = os.path.join(image_dir, image_filename)

    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_filename}' does not exist in the 'augmented' directory.")
    else:
        # Make a prediction and compare with the actual values
        prediction, actual_a, actual_b = predict(image_path)
        
        # Print the comparison
        print(f"Image: {image_filename}")
        print(f"Predicted mean chrominance: a* = {prediction[0][0]:.4f}, b* = {prediction[0][1]:.4f}")
        print(f"Actual mean chrominance: a* = {actual_a:.4f}, b* = {actual_b:.4f}")
