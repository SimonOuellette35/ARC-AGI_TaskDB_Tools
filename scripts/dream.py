import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dreaming.dreaming_data_generator import DreamingDataGenerator


gen = DreamingDataGenerator()

from collections import Counter

counter = Counter()

print("Generating...")
combined_tasks = gen.dream(50)
print("Done")