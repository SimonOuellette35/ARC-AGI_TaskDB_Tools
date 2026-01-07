from datasets.dreaming_data_generator import DreamingDataGenerator
from datasets.grid_example_generator import generate_grid_examples
import ARC_gym.utils.visualization as viz


gen = DreamingDataGenerator()

from collections import Counter

counter = Counter()

print("Generating...")
combined_tasks = gen.dream(50)
print("Done")

# for combined_task in combined_tasks:
#     print(f"==> combined_task: {combined_task['name']}")

    # print(f"==> Program:")
    # print(f"{combined_task['program']}")

    # # Generate 3 input-output grid examples
    # instructions = combined_task.get('instructions', [])
    # grid_categories = combined_task.get('grid_categories', ['basic'])
    # if not isinstance(grid_categories, list):
    #     grid_categories = ['basic']

    # parameter_tags = []
    # task_params = combined_task.get('parameters') or combined_task.get('parameter', [])
    # if isinstance(task_params, list):
    #     parameter_tags = list(task_params)

    # min_grid_dim = combined_task.get('min_grid_dim')
    # max_grid_dim = combined_task.get('max_grid_dim')

    # examples = generate_grid_examples(
    #     instructions,
    #     num_examples=3,
    #     grid_categories=grid_categories,
    #     strict=True,
    #     parameters=parameter_tags if parameter_tags else None,
    #     min_grid_dim=min_grid_dim,
    #     max_grid_dim=max_grid_dim
    # )

    # print(f"\n==> Generated {len(examples)} examples:")
    # for i, example in enumerate(examples, 1):
    #     print(f"\nExample {i}:")
    #     viz.draw_grid_pair(example.get('input', []), example.get('output', []))
