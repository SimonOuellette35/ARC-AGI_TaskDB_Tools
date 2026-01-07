# ARC-AGI Task DB Management, Curation & Generation Tools

![Main Diagram](images/main_diagram.png)

This is a framework that allows maintaining a database of ARC-AGI tasks (for training purposes), via a web-based UI.

It also includes a script to generate training examples from this task database, and a "dreaming" framework that uses evolutionary algorithm-like operators (crossover, mutation, composition) to derive automatically new tasks from existing tasks. These automatically generated tasks can then be reviewed via the UI, and either deleted if they don't seem to make sense, or validated if the user likes the new task.

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

## Example 2: task generation

## Example 3: training/validation data generation

