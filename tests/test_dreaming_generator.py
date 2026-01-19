from dreaming.dreaming_data_generator import DreamingDataGenerator
from dreaming.grid_example_generator import generate_grid_examples


def test_compare_two_tasks():

    def run_test(ex_task1, ex_task2):
        # Generate examples
        examples = generate_grid_examples(
            ex_task1['instructions'],
            num_examples=25,
            grid_categories=ex_task1['grid_categories'],
            strict=False,
            parameters=ex_task1['parameters'],
            min_grid_dim=min_grid_dim,
            max_grid_dim=max_grid_dim,
            catch_exceptions=False
        )

        tasks_to_mark_obsolete = []
        return gen._compare_two_tasks(ex_task1, ex_task2, examples, tasks_to_mark_obsolete, 0)

    gen = DreamingDataGenerator()
    min_grid_dim = 6
    max_grid_dim = 20

    # Example #1: two distinct tasks
    task1 = {
        "name": "Draw Arbitrary Points Upper Left V4",
        "program": "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0),\n  set_pixels(N+0, 2, 0, \"param2\"),\n  del(N+0),\n  set_pixels(N+0, 0, 2, \"param2\"),\n  del(N+0),\n  set_pixels(N+0, 2, 2, \"param1\"),\n  del(N+0)\n]",
        "instructions": [
            [0, 38, 1, 61, 2, 4, 2, 4, 2, "param1", 3],
            [0, 51, 1, 61, 3],
            [0, 38, 1, 61, 2, 6, 2, 4, 2, "param2", 3],
            [0, 51, 1, 61, 3],
            [0, 38, 1, 61, 2, 4, 2, 6, 2, "param2", 3],
            [0, 51, 1, 61, 3],
            [0, 38, 1, 61, 2, 6, 2, 6, 2, "param1", 3],
            [0, 51, 1, 61, 3]
        ],
        "comments": [],
        "curriculum_level": 0,
        "source": "Auto",
        "parameters": [
            "color",
            "color"
        ],
        "grid_categories": [
            "distinct_colors_adjacent",
            "distinct_colors_adjacent_empty",
            "non_symmetrical_shapes",
            "simple_filled_rectangles"
        ],
        "min_grid_dim": 4,
        "max_grid_dim": 30,
        "validated": True
    }

    task2 = {
        "name": "Shift Down + Draw Upper Border",
        "program": "[\n  sub(0, 1),\n  set_pixels(N+0, N+1, N+1, \"param1\"),\n  del(N+0),\n  set_pixels(N+1, N+1.x, N+0, \"param1\"),\n  del(N+1),\n  del(N+0)\n]",
        "instructions": [
            [0, 25, 1, 4, 2, 5, 3],
            [0, 38, 1, 61, 2, 62, 2, 62, 2, "param1", 3],
            [0, 51, 1, 61, 3],
            [0, 38, 1, 62, 2, 62, 52, 2, 61, 2, "param1", 3],
            [0, 51, 1, 62, 3],
            [0, 51, 1, 61, 3]
        ],
        "comments": [],
        "curriculum_level": 1,
        "source": "Auto",
        "parameters": [
            "fg_color"
        ],
        "grid_categories": [
            "distinct_colors_adjacent_empty_fill",
            "basic",
            "non_symmetrical_shapes",
            "shearable_grids",
            "simple_filled_rectangles"
        ],
        "min_grid_dim": 4,
        "max_grid_dim": 30,
        "validated": True
    }

    is_same = run_test(task1, task2)

    assert is_same is False

    # Example #2: two distinct tasks

    task1 = {
    "program": "[\n  add(N+0.x, 1),\n  set_pixels(N+0, N+1, N+0.y, N+0.c),\n  del(N+1),\n  del(N+0),\n  set_pixels(N+0, 0, N+0.y, 0),\n  del(N+0),\n  crop(N+0, 0, 0, N+0.max_x, N+0.height),\n  del(N+0),\n  equal(N+0.c, param1),\n  equal(N+0.c, param2),\n  switch(N+1, N+2, param2, param1, N+0.c),\n  del(N+1),\n  del(N+1),\n  set_pixels(N+0, N+0.x, N+0.y, N+1),\n  del(N+0),\n  del(N+0)\n]",
    "instructions": [
      [0, 24, 1, 61, 52, 2, 5, 3],
      [0, 38, 1, 61, 2, 62, 2, 61, 53, 2, 61, 54, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 61, 3],
      [0, 38, 1, 61, 2, 4, 2, 61, 53, 2, 4, 3],
      [0, 51, 1, 61, 3],
      [0, 36, 1, 61, 2, 4, 2, 4, 2, 61, 55, 2, 61, 58, 3],
      [0, 51, 1, 61, 3],
      [0, 18, 1, 61, 54, 2, "param1", 3],
      [0, 18, 1, 61, 54, 2, "param2", 3],
      null,
      [0, 51, 1, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 38, 1, 61, 2, 61, 52, 2, 61, 53, 2, 62, 3],
      [0, 51, 1, 61, 3],
      [0, 51, 1, 61, 3]
    ],
    "comments": [
      "x + 1",
      "Grid",
      "",
      "",
      "Grid",
      "",
      "Grid",
      "",
      "for each pixel, whether it has color 0",
      "for each pixel, whether it has color 1",
      "0 pixels set to 1, 1 pixels set to 0",
      "",
      "",
      "Grid",
      "",
      ""
    ],
    "curriculum_level": 1,
    "name": "Shift Right + SwapColor",
    "source": "Manual",
    "parameters": [
      "existing_color",
      "existing_color"
    ],
    "grid_categories": [
      "basic"
    ],
    "min_grid_dim": 3,
    "max_grid_dim": 30,
    "validated": True
  }

  task2 = {
    "program": "[\n  sub(N+0.y, 1),\n  set_pixels(N+0, N+0.x, N+1, N+0.c),\n  del(N+1),\n  del(N+0),\n  set_pixels(N+0, N+0.x, N+0.max_y, 0),\n  del(N+0),\n  crop(N+0, 0, 1, N+0.width, N+0.height),\n  del(N+0),\n  equal(N+0.c, param1),\n  switch(N+1, param1, param2),\n  del(N+1),\n  set_pixels(N+0, N+0.x, N+0.y, N+1),\n  del(N+0),\n  del(N+0)\n]",
    "instructions": [
      [0, 25, 1, 61, 53, 2, 5, 3],
      [0, 38, 1, 61, 2, 61, 52, 2, 62, 2, 61, 54, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 61, 3],
      [0, 38, 1, 61, 2, 61, 52, 2, 61, 56, 2, 4, 3],
      [0, 51, 1, 61, 3],
      [0, 36, 1, 61, 2, 4, 2, 5, 2, 61, 57, 2, 61, 58, 3],
      [0, 51, 1, 61, 3],
      [0, 18, 1, 61, 54, 2, "param1", 3],
      [0, 21, 1, 62, 2, "param1", 2, "param2", 3],
      [0, 51, 1, 62, 3],
      [0, 38, 1, 61, 2, 61, 52, 2, 61, 53, 2, 62, 3],
      [0, 51, 1, 61, 3],
      [0, 51, 1, 61, 3]
    ],
    "comments": [
      "y - 1",
      "Grid",
      "",
      "",
      "Grid",
      "",
      "Grid",
      "",
      "for each pixel, whether it has color 0",
      "non-0 color pixels set to 1",
      "Grid",
      "",
      ""
    ],
    "curriculum_level": 1,
    "name": "Shift Up + OtherColors",
    "source": "Manual",
    "parameters": [
      "existing_color",
      "color"
    ],
    "grid_categories": [
      "basic"
    ],
    "min_grid_dim": 3,
    "max_grid_dim": 30,
    "validated": True
  },


    is_same = run_test(task1, task2)

    assert is_same is False

    # Example #3: two distinct tasks

    is_same = run_test(task1, task2)

    assert is_same is False

    # Example #4: two identical tasks (but slightly different code)

    is_same = run_test(task1, task2)

    assert is_same is True

    # Example #5: two identical tasks (but slightly different code)

    is_same = run_test(task1, task2)

    assert is_same is True

    # Example #6: two identical tasks (but slightly different code)

    is_same = run_test(task1, task2)

    assert is_same is True
