import cv2
import argparse
import os
from dense_crowd_ai import DensityEstimator

def main():
    parser = argparse.ArgumentParser(description="Test Crowd Density Heatmap Engine")
    parser.add_argument('--image', type=str, required=True, help="Path to test image")
    parser.add_argument('--weights', type=str, default='model_best.pth.tar', help="Path to model weights")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found.")
        print("Please download the CSRNet weights and place them in the directory.")
        return

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    print("Loading model...")
    estimator = DensityEstimator(args.weights)
    
    print("Reading image...")
    frame = cv2.imread(args.image)
    if frame is None:
        print("Failed to read image.")
        return

    print("Processing frame...")
    result = estimator.process_frame(frame)
    
    count = result['estimated_count']
    heatmap = result['heatmap_frame']
    
    print("\n====================================")
    print(f" Estimated Crowd Count: {count}")
    print("====================================\n")
    
    output_path = "output_heatmap.jpg"
    cv2.imwrite(output_path, heatmap)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    main()
