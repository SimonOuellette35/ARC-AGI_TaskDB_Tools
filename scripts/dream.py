import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dreaming.dreaming_data_generator import DreamingDataGenerator


gen = DreamingDataGenerator()

from collections import Counter

counter = Counter()

parser = argparse.ArgumentParser()
parser.add_argument('N', type=int, nargs='?', default=50, help='Parameter value to pass to gen.dream() (default: 50)')
args = parser.parse_args()

print("Generating...")
combined_tasks = gen.dream(args.N)
print("Done")