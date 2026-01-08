import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dreaming.dreaming_data_generator import DreamingDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate training and validation data")
    parser.add_argument("--trainN", type=int, default=250000, help="Number of training samples per curriculum level (default: 250000)")
    parser.add_argument("--valN", type=int, default=10000, help="Number of validation samples (default: 10000)")
    parser.add_argument("--skip_training", action="store_true", default=False, help="Skip training data generation (default: False)")
    parser.add_argument("--skip_validation", action="store_true", default=False, help="Skip validation data generation (default: False)")
    parser.add_argument("--curriculum_lvl", type=int, default=None, help="Curriculum level to generate (default: None, generates all levels)")
    parser.add_argument("--mixed_mode", action="store_true", default=False, help="Generate samples from all tasks regardless of curriculum level (default: False)")
    
    args = parser.parse_args()
    
    trainN = args.trainN
    valN = args.valN

    generator = DreamingDataGenerator()

    # Generate training data
    if not args.skip_training:
        if args.mixed_mode:
            print(f"Generating {trainN} training samples in mixed mode (all curriculum levels)...")
        elif args.curriculum_lvl is not None:
            print(f"Generating {trainN} training samples for curriculum level {args.curriculum_lvl}...")
        else:
            print(f"Generating {trainN} training samples per curriculum level...")
        generator.generate(trainN, "training", args.curriculum_lvl, args.mixed_mode)

    # Generate validation data
    if not args.skip_validation:
        if args.mixed_mode:
            print(f"Generating {valN} validation samples in mixed mode (all curriculum levels)...")
        elif args.curriculum_lvl is not None:
            print(f"Generating {valN} validation samples for curriculum level {args.curriculum_lvl}...")
        else:
            print(f"Generating {valN} validation samples per curriculum level...")
        generator.generate(valN, "validation", args.curriculum_lvl, args.mixed_mode)


if __name__ == "__main__":
    main()
