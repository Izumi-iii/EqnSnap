import argparse
from pathlib import Path

from PIL import Image
from pix2tex.cli import LatexOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pix2tex with an equation image.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to an equation image")
    args = parser.parse_args()

    print("Loading pix2tex model...")
    model = LatexOCR()
    print("Model loaded successfully.")

    if args.image is None:
        print("Pass an image path to run recognition.")
        return

    with Image.open(args.image) as image:
        latex = model(image.convert("RGB"))

    print("\nRecognized LaTeX:")
    print(latex)


if __name__ == "__main__":
    main()
