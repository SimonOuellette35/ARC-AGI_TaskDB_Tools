from dreaming.dreaming_data_generator import DreamingDataGenerator, block_of_text_to_program_lines
from dreaming.grid_example_generator import generate_grid_examples
from AmotizedDSL.prog_utils import ProgUtils


def test_compare_two_tasks():

    def run_test(ex_task1, ex_task2):
        prog1 = block_of_text_to_program_lines(ex_task1['program'])
        instructions = ProgUtils.convert_user_format_to_token_seq(prog1)

        # Generate examples
        examples = generate_grid_examples(
            instructions,
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
    }

    is_same = run_test(task1, task2)

    assert is_same is False

    # Example #3: two distinct tasks

    task1 = {
        "program": "[\n  add(N+0.width, N+0.x),\n  set_pixels(N+0, N+1, N+0.y, N+0.c),\n  del(N+0),\n  del(N+0)\n]",
        "comments": [
        "Add grid width to its x values",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "name": "Tile Right",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "basic"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 15,
        "validated": True
    }

    task2 = {
        "program": "[\n  add(N+0.height, N+0.y),\n  set_pixels(N+0, N+0.x, N+1, N+0.c),\n  del(N+0),\n  del(N+0)\n]",
        "comments": [
        "Add grid height to its y values",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "name": "Tile Down",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "basic"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 15,
        "validated": True
    }

    is_same = run_test(task1, task2)

    assert is_same is False

    # Example #4: two identical tasks (but slightly different code)
    task1 = {
        "name": "Draw Diagonal",
        "program": "[\n  and(N+0.max_y, N+0.x),\n  set_pixels(N+0, N+0.x, N+1, param1),\n  del(N+0),\n  del(N+0)\n]",
        "comments": [
        "grid height / 2",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "source": "Auto",
        "parameters": [
        "fg_color"
        ],
        "grid_categories": [
        "basic"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }

    task2 = {
        "name": "Draw Diagonal #2",
        "program": "[\n  and(N+0.max_y, N+0.x),\n  set_pixels(N+0, N+0.x, N+1, 3),\n  set_pixels(N+0, N+0.x, N+1, param1),\n  del(N+2),\n  del(N+0),\n  del(N+0)\n]",
        "instructions": [
        ],
        "comments": [
        "grid height / 2",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "source": "Auto",
        "parameters": [
        "fg_color"
        ],
        "grid_categories": [
        "basic"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }
    
    is_same = run_test(task1, task2)

    assert is_same is True

    # Example #5: two identical tasks (but slightly different code)
    task1 =   {
        "program": "[\n  get_objects(N+0),\n  del(N+0),\n  index(N+0, 0),\n  del(N+0)\n]",
        "comments": [
        "list of objects",
        "",
        "Grid",
        ""
        ],
        "curriculum_level": 0,
        "name": "Object Crop",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "single_object",
        "single_object_noisy_bg"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }


    task2 = {
        "program": "[\n  get_objects(N+0),\n  del(N+0),\n  equal(N+0.c, 2),\n  equal(N+0.c, 3),\n  or(N+1, N+2),\n  index(N+0, 0),\n  del(N+1),\n  del(N+1),\n  del(N+1),\n  del(N+0)\n]",
        "instructions": [
        ],
        "comments": [
        "list of objects",
        "",
        "Grid",
        ""
        ],
        "curriculum_level": 0,
        "name": "Object Crop #2",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "single_object",
        "single_object_noisy_bg"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }

    is_same = run_test(task1, task2)

    assert is_same is True

    # Example #6: two identical tasks (but slightly different code)
    task1 =   {
        "program": "[\n  get_objects(N+0),\n  get_bg(N+0),\n  del(N+0),\n  sub(N+0.max_x, N+0.x),\n  set_x(N+0, N+2),\n  del(N+0),\n  del(N+1),\n  rebuild_grid(N+0, N+1),\n  del(N+0),\n  del(N+0)\n]",
        "comments": [
        "list of objects",
        "grid background",
        "",
        "max_x - x",
        "list of objects",
        "",
        "",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "name": "Object Flip H",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "non_symmetrical_shapes"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }

    task2 =   {
        "program": "[\n  get_objects(N+0),\n  get_bg(N+0),\n  del(N+0),\n  sub(N+0.max_x, N+0.x),\n  set_x(N+0, N+2),\n  set_y(N+0, N+0.y),\n  del(N+0),\n  del(N+1),\n  rebuild_grid(N+0, N+1),\n  del(N+2),\n  del(N+0),\n  del(N+0)\n]",
        "comments": [
        "list of objects",
        "grid background",
        "",
        "max_x - x",
        "list of objects",
        "",
        "",
        "Grid",
        "",
        ""
        ],
        "curriculum_level": 0,
        "name": "Object Flip H #2",
        "source": "Manual",
        "parameters": [],
        "grid_categories": [
        "non_symmetrical_shapes"
        ],
        "min_grid_dim": 3,
        "max_grid_dim": 30,
        "validated": True
    }

    is_same = run_test(task1, task2)

    assert is_same is True
