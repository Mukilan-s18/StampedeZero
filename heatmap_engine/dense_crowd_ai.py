import cv2
import torch
import numpy as np
from torchvision import transforms
from model import CSRNet

class DensityEstimator:
    def __init__(self, weight_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CSRNet(load_weights=True)
        
        # The map_location parameter prevents CUDA crashes on laptops
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        # Handle dicts vs raw state dicts
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval() # CRITICAL: Turns off dropout/batchnorm for prediction
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        # 1. Preprocess OpenCV frame to PIL -> Tensor
        # Convert from BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply standard PyTorch transforms
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # 2. Run inference
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # 3. Calculate count: The sum of all values in the density map
        count = int(torch.sum(output).item())
        
        # 4. Generate the Heatmap Overlay using cv2.addWeighted()
        density_map = output.squeeze().cpu().numpy()
        
        # CNN outputs are usually 1/8th the size. Resize back up to original image dims.
        original_h, original_w = frame.shape[:2]
        density_map_resized = cv2.resize(density_map, (original_w, original_h))
        
        # Normalize for Colors (0-255 scale)
        d_min = np.min(density_map_resized)
        d_max = np.max(density_map_resized)
        if d_max - d_min > 0:
            density_map_normalized = (density_map_resized - d_min) / (d_max - d_min) * 255.0
        else:
            density_map_normalized = np.zeros_like(density_map_resized)
            
        density_map_normalized = density_map_normalized.astype(np.uint8)
        
        # Apply the glow effect
        heatmap = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)
        
        # Merge it with the original frame (so the background is visible through the glow)
        final_overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        return {
            "estimated_count": count,
            "heatmap_frame": final_overlay # NumPy array
        }
