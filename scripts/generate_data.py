import config
from datasets.dreaming_data_generator import DreamingDataGenerator


trainN = 200000
valN = 1000
cfg = config.load_gridcoder_config()

generator = DreamingDataGenerator()

print("cfg = ", cfg)

# Generate training data
print(f"Generating {trainN} training samples per curriculum level...")
training_data = generator.generate(trainN, "training")

print("Training data saved to training_<curriculum_level>.json")

# Generate validation data
print(f"Generating {valN} validation samples...")
validation_data = generator.generate(valN, "validation")

print("Validation data saved to validation_<curriculum_level>.json")
