# ARC-AGI Task DB Management, Curation & Generation Tools

![Main Diagram](images/main_diagram.png)

This is a framework to help maintain, review, and curate a database of ARC-AGI tasks (for training purposes), via a web-based UI.

It also includes a script to generate training examples from this task database, and a "dreaming" framework that uses evolutionary algorithm-like operators (crossover, mutation, composition) to derive automatically new tasks from existing tasks. Think of it like data augmentation for your training tasks. These automatically generated tasks can then be reviewed via the UI, and either deleted if they don't seem to make sense, or validated if the user likes the new task.

### There are 3 main use cases / tools:

1. The task DB manager UI (task_manager.html/css)
2. The task generation script
3. The training/validation data generation script

# How to use:
## Install

1. Install ARC_gym (used to generate grids that follow various constraints/rules)

    ```
    git clone https://github.com/SimonOuellette35/ARC_gym.git
    cd ARC_gym
    pip install -e .
    ```

2. Add ARC-AGI-2 training data grids to ARC_gym local repo:

    ```
    cd ARC_gym
    git clone https://github.com/arcprize/ARC-AGI-2.git
    ```

3. Install AmotizedDSL (the acutal DSL used by the program ground truths + program interpreter and utilities)

    ```
    git clone https://github.com/SimonOuellette35/AmotizedDSL.git
    cd AmotizedDSL
    pip install -e .
    ```

## Example 1: task DB manager

    python manager_ui/task_DB_manager.py
    In your browser, go to URL: http://localhost:8000/manager_ui/task_manager.html
    

You will be a UI that lists the content of the task_DB.json file:

![Screenshot 1](images/screenshot1.jpg)

Click a task on the left (Example: Crop Inside) opens a panel with 3 input-output grid examples for this task. You can click the "refresh" icon to the right of the title "GRID EXAMPLES" to re-generate new examples if desired. If you scroll down, you will see the program ground truth written in the [AmotizedDSL](https://github.com/SimonOuellette35/AmotizedDSL) custom DSL. It is possible to specify comments for manually created tasks. This is optional, and not supported for automatically created tasks.

![Screenshot 2](images/screenshot2.jpg)

The "INSTRUCTIONS" section is a representation of the same program in token sequence format (see [AmotizedDSL](https://github.com/SimonOuellette35/AmotizedDSL) documentation for further detail) -- much less user friendly, you do not need to edit or write this part when you create a new task, as it is automatically generated from the text-based program code in the "PROGRAM (WITH COMMENTS)" section.

There is the "PARAMETERS" section, that lists the parameter type of each parameter in the program ground truth. See the section below about "Program Parameters" for more information. Next to is is the "GRID CATEGORIES", which informs [ARC gym](https://github.com/SimonOuellette35/ARC_gym) what rules to use to generate the input grids (see the respective documentation in ARC gym for more information). Finally, the "GRID DIMENSIONS" is the minimum and maximum recommended grid dimensions when generating the input grid. This is fed to ARC gym as well, but note that at the moment it is not always respected for a variety of reasons (bugs + for grid dimensions are not sensible for certain types of grid categories, etc.)

More details below on the EDIT, DELETE, ADD ENTRY functionalities.

## Example 2: task generation via the "Dreaming" framework

    python scripts/dream.py 50

This runs the dreaming framework to generate 50 new programs and adds them to task_DB.json with status "Unvalidated" (red question mark icon in the task list).

## Example 3: training/validation data generation

You can use task_DB.json to generate training and validation files for model training/evaluation.

    python scripts/generate_data.py --trainN 10000 --valN 1000 --curriculum_lvl 0
    
This will generate training_0.json containing 10000 task samples taken randomly from the curriculum level 0 tasks in task_DB.json, and validation_0.json containing 1000 different task samples.

    python scripts/generate_data.py --trainN 10000 --valN 1000
    
This will generate a training_<curriculum_lvl>.json and validation_<curriculum_lvl>.json for each curriculum level that exists in task_DB.json.

    python scripts/generate_data.py --trainN 10000 --valN 1000 --mixed_mode
    
Generates a training.json and validation.json of task samples from any curriculum level, all in one file (good if you don't plan to do any curriculum learning).

### Using the UI to create/edit/delete tasks

## Adding a new task

Click "ADD ENTRY" on top of the left panel. This will create a new empty form on the main panel. Note that this is an advanced feature that assumes you are familiar with the [AmotizedDSL](https://github.com/SimonOuellette35/AmotizedDSL).

![Screenshot 3](images/screenshot3.jpg)

You can give an arbitrary task name and a curriculum level. The curriculum level is somewhat subjective and depends on how you can to structure the training process (or if you care at all about the concept of curriculum) -- but for a brand new concept task I, not composed or some "crossover" from two other tasks, I suggest a value of 0.

The program writing part is explained further in the next section "Editing an existing task".

If your task contains parameters, you can add their respective types via the parameters drop-down list. You must also specify a "grid categories" tag -- at least 1, up to 5.

Finally, you simply push the Save button to write this new task to the DB (or Cancel otherwise).

## Editing an existing task

## Program Parameters
