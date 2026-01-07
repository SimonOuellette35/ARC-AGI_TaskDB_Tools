# ARC-AGI Task DB Management, Curation & Generation Tools

![Main Diagram](images/main_diagram.png)

This is a framework that allows maintaining a database of ARC-AGI tasks (for training purposes), via a web-based UI.

It also includes a script to generate training examples from this task database, and a "dreaming" framework that uses evolutionary algorithm-like operators (crossover, mutation, composition) to derive automatically new tasks from existing tasks. These automatically generated tasks can then be reviewed via the UI, and either deleted if they don't seem to make sense, or validated if the user likes the new task.

### There are 4 main use cases / tools:

1. The task DB manager UI (task_manager.html/css)
2. The task generation script
3. The training/validation data generation script
4. Custom training/validation/data generation loop via the CurriculumDataset class
