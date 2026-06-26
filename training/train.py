from numpy.lib.twodim_base import fliplr
from sympy.printing.pytorch import torch
from ultralytics import YOLO
import numpy as np
import os
import sys

def main():
    try:
        combined_dog_yaml = "dog_attributes.yaml"

        if not os.path.exists(combined_dog_yaml):
            print(f"Error: Dataset YAML file {combined_dog_yaml} not found.")
            sys.exit(1)

        model = YOLO("yolov8n-pose.pt")

        model.train(
        data=combined_dog_yaml,
        epochs=150,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        batch=8,
        optimizer="Adam",
        lr0=0.0015,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.3,
        translate=0.05,
        flipud=0.0,
        fliplr=0.5,
        patience=20
        )


        results = model("data/test/images")

        for i, result in enumerate(results):
            result.save(filename=f"runs/detect/output_{i}.jpg")

    except Exception as e:
        print(f"An error has occurred: {e}")
        sys.exit(1)



if __name__ == "__main__":
    main()
