# ARC-AGI_TaskDB_Tools
ARC-AGI Task database manager UI &amp; "Dreaming" framework

![Main Diagram](images/main_diagram.png)

This is a framework that allows maintaining a database of ARC-AGI tasks (for training purposes), via a web-based UI.

It also includes a script to generate training examples from this task database, and a "dreaming" framework that uses evolutionary algorithm-like operators (crossover, mutation, composition) to derive automatically new tasks from existing tasks. These automatically generated tasks can then be reviewed via the UI, and either deleted if they don't seem to make sense, or validated if the user likes the new task.