"""
Data generator for "dreaming" - generating task examples from task_DB.json.
"""

import json
from tqdm import tqdm
import random
import math
import sys
import re
import ast
from pathlib import Path
import numpy as np
import inspect
from dreaming.grid_example_generator import generate_grid_examples, replace_parameter_placeholders
import AmotizedDSL.DSL as DSL
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import copy
from dreaming.utils import DreamingUtils
import ARC_gym.utils.visualization as viz


VERBOSE = False
DISPLAY_PROGRAM_VALIDATION_TRACEBACK = False

# ============================================================================ Public methods ===============================================================================

def block_of_text_to_program_lines(program_str):
    """Take a block of text from a task DB program description and convert it into a list of
    user format instruction strings.
    
    Args:
        program_str: String like "[\n  get_objects(N+0),\n  del(N+0),\n  ...\n]"
    
    Returns:
        List of instruction strings like ["get_objects(N+0)", "del(N+0)", ...]
    """
    program_str = program_str.strip()
    if program_str.startswith('['):
        program_str = program_str[1:]
    if program_str.endswith(']'):
        program_str = program_str[:-1]
    
    lines = [line.strip() for line in program_str.split('\n') if line.strip() and line.strip() not in ['[', ']']]
    
    instructions = []
    for line in lines:
        line = line.rstrip(',').strip()
        if line:
            instructions.append(line)
    
    return instructions

def program_lines_to_block_of_text(instruction_strings):
    """Convert a list of user format instruction strings back to a single block of text to be saved in Task DB.
    
    Args:
        instruction_strings: List of instruction strings
    
    Returns:
        Program string in the format "[\n  instruction1,\n  instruction2,\n  ...\n]"
    """
    if not instruction_strings:
        return "[]"
    
    formatted = "[\n"
    for i, instr in enumerate(instruction_strings):
        formatted += f"  {instr}"
        if i < len(instruction_strings) - 1:
            formatted += ","
        formatted += "\n"
    formatted += "]"
    return formatted


class DreamingDataGenerator:

    def __init__(self):
        self._task_db_cache = None

    def dream(self, N, probs=[0.05, 0.4], max_step_count=20, return_only=False):
        """Generate new tasks in the task_DB by combining two existing tasks."""

        new_tasks = []

        if not hasattr(self, 'task_DB') or not self.task_DB:
            self.task_DB = self.sample_task_DB()

        for task_num in range(N):
            print(f"==> Generating task #{task_num+1}")
            valid = False
            while not valid:
                new_task = self.dream_iteration(probs, max_step_count, return_only)

                if new_task is None:
                    continue
                else:
                    valid = True
                    new_tasks.append(new_task)
            
            print(f"==> Successfully generateda a new task: {new_task['name']}")
            print(f"==> Corresponding program: {new_task['program']}")

        return new_tasks

    def generate(self, N, basename, curriculum_lvl_arg=None, mixed_mode=False):
        """Generate N task examples and save to JSON file.
        
        Args:
            N: Number of task examples to generate for each curriculum level (or overall if mixed_mode=True)
            basename: Output JSON basename (_<curriculum level>.json gets appended to it, or .json if mixed_mode=True)
            curriculum_lvl_arg: Optional curriculum level to generate. If None, generates all levels.
            mixed_mode: If True, generates N examples overall from all tasks regardless of curriculum level.
        """
        if mixed_mode:
            # Generate N examples overall from all tasks, ignoring curriculum levels
            print(f"==> Generating {N} examples in mixed mode (all curriculum levels)")
            tasks = self.sample_task_DB()
            
            if len(tasks) == 0:
                print("No tasks found in task_DB")
                return
            
            samples = self.generate_samples(tasks, N)
            
            # Save to JSON file with integer lists on single lines
            current_filename = f'{basename}.json'
            with open(current_filename, 'w') as f:
                formatted_json = format_json_with_compact_integer_lists(samples, indent=2)
                f.write(formatted_json)
            
            print(f"Successfully generated {len(samples)} task examples and saved to {current_filename}")
            return
        
        curriculum_lvl = 0
        if curriculum_lvl_arg is not None:
            curriculum_lvl = curriculum_lvl_arg

        while True:
            print(f"==> Generating data for curriculum level {curriculum_lvl}")
            tasks = self.sample_task_DB(curriculum_lvl)

            if len(tasks) == 0:
                return

            samples = self.generate_samples(tasks, N)
            
            # Save to JSON file with integer lists on single lines
            current_filename = f'{basename}_{curriculum_lvl}.json'
            with open(current_filename, 'w') as f:
                formatted_json = format_json_with_compact_integer_lists(samples, indent=2)
                f.write(formatted_json)
            
            print(f"Successfully generated {len(samples)} task examples and saved to {current_filename}")

            if curriculum_lvl_arg is not None:
                break
            
            curriculum_lvl += 1

    def generate_incremental(self, N, basename, task_list):
        """Generate N task examples overall, split uniformly among tasks in task_list, and append to JSON file.
        
        Args:
            N: Total number of task examples to generate overall
            basename: Output JSON basename (e.g., 'training' -> 'training.json')
            task_list: List of task names to generate samples for
        """
        if not task_list:
            print("==> task_list is empty, nothing to generate")
            return
        
        # Load task_DB
        all_tasks = self.sample_task_DB()
        
        # Create a mapping from task name to task dict
        task_dict = {}
        for task in all_tasks:
            task_name = task.get('name', '')
            if task_name:
                task_dict[task_name] = task
        
        # Filter task_list to only include tasks that exist in task_DB
        valid_tasks = []
        for task_name in task_list:
            if task_name in task_dict:
                valid_tasks.append(task_dict[task_name])
            else:
                print(f"==> Warning: Task '{task_name}' not found in task_DB, skipping...")
        
        if not valid_tasks:
            print("==> No valid tasks found in task_DB, nothing to generate")
            return
        
        # Calculate samples per task (split uniformly)
        samples_per_task = N // len(valid_tasks)
        remainder = N % len(valid_tasks)
        
        # Generate samples for each task
        all_samples = []
        for i, task in enumerate(valid_tasks):
            # Distribute remainder samples to first few tasks
            num_samples = samples_per_task + (1 if i < remainder else 0)
            
            if num_samples == 0:
                continue
            
            print(f"==> Generating {num_samples} samples for task '{task.get('name', '')}'")
            
            # Generate samples for this specific task
            task_samples = []
            for _ in tqdm(range(num_samples), desc=f"Generating for {task.get('name', '')}"):
                # Get task properties
                instructions = task.get('instructions', [])
                if not instructions:
                    print(f"Warning: Task '{task.get('name', '')}' has no instructions, skipping...")
                    continue
                
                grid_categories = task.get('grid_categories', ['basic'])
                if not isinstance(grid_categories, list):
                    grid_categories = ['basic']
                
                parameter_tags = []
                task_params = task.get('parameters') or task.get('parameter')
                if isinstance(task_params, list):
                    parameter_tags = list(task_params)
                
                min_grid_dim = task.get('min_grid_dim')
                max_grid_dim = task.get('max_grid_dim')
                
                # Generate 3 input-output examples for this task
                valid = False
                while not valid:
                    try:
                        examples = generate_grid_examples(
                            instructions, 
                            num_examples=3, 
                            grid_categories=grid_categories,
                            strict=False,
                            parameters=parameter_tags,
                            min_grid_dim=min_grid_dim,
                            max_grid_dim=max_grid_dim
                        )
                        
                        if len(examples) < 3:
                            continue
                        
                        examples = examples[:3]
                        
                        # Remove 'parameters' field and ensure 'object_mask' is present
                        for example in examples:
                            if 'parameters' in example:
                                del example['parameters']
                            if 'object_mask' not in example:
                                example['object_mask'] = []
                        
                        # Create entry in the same format as validation.json
                        entry = {
                            'train': examples,
                            'test': [],
                            'prog': instructions,
                            'name': task.get('name', '')
                        }
                        
                        task_samples.append(entry)
                        valid = True
                        
                    except Exception as e:
                        continue
            
            all_samples.extend(task_samples)
        
        if not all_samples:
            print("==> No samples were generated")
            return
        
        # Load existing JSON file if it exists and append new samples
        filename = f'{basename}.json'
        existing_samples = []
        try:
            with open(filename, 'r') as f:
                existing_samples = json.load(f)
                if not isinstance(existing_samples, list):
                    existing_samples = []
        except FileNotFoundError:
            existing_samples = []
        except json.JSONDecodeError:
            print(f"==> Warning: {filename} exists but is not valid JSON, will overwrite")
            existing_samples = []
        
        # Append new samples to existing ones
        combined_samples = existing_samples + all_samples
        
        # Write back to file (preserving existing data + new data)
        with open(filename, 'r+') as f:
            formatted_json = format_json_with_compact_integer_lists(combined_samples, indent=2)
            f.seek(0)
            f.write(formatted_json)
            f.truncate()
        
        print(f"Successfully generated {len(all_samples)} new task examples and appended to {filename} (total: {len(combined_samples)})")

# =========================================================================== Private methods ==============================================================================


    def sample_task_DB(self, curriculum_lvl=None):
        # Load task_DB.json (cached - only load once, reuse across curriculum levels)
        if self._task_db_cache is None:
            try:
                with open('task_DB.json', 'r') as f:
                    self._task_db_cache = json.load(f)
            except FileNotFoundError:
                raise ValueError("task_DB.json not found")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in task_DB.json: {e}")

        if not self._task_db_cache:
            raise ValueError("task_DB.json is empty")

        # Filter tasks: exclude tasks with validated == False
        # Also filter by curriculum_lvl if specified
        filtered_tasks = []
        for task in self._task_db_cache:
            # Exclude tasks with validated == False
            if task.get("validated") == False:
                continue
            # Filter by curriculum_lvl if specified
            if curriculum_lvl is not None:
                # If 'curriculum_level' missing, treat as None and filter out
                if "curriculum_level" in task and task["curriculum_level"] == curriculum_lvl:
                    filtered_tasks.append(task)
            else:
                filtered_tasks.append(task)
        
        return filtered_tasks       

    def generate_samples(self, tasks, N):
        # Generate examples for each selected task
        samples = []
        
        for _ in tqdm(range(N), desc='Generating samples'):
            task_idx = random.randint(0, len(tasks) - 1)
            task = tasks[task_idx]

            # Get task properties
            instructions = task.get('instructions', [])
            if not instructions:
                print(f"Warning: Task {task_idx + 1} has no instructions, skipping...")
                continue
            
            grid_categories = task.get('grid_categories', ['basic'])
            if not isinstance(grid_categories, list):
                grid_categories = ['basic']
            
            # Just use parameters field from task directly (should already be a list of tags)
            parameter_tags = []
            task_params = task.get('parameters') or task.get('parameter')
            if isinstance(task_params, list):
                parameter_tags = list(task_params)
            else:
                parameter_tags = []
            
            # Get min_grid_dim and max_grid_dim from task
            min_grid_dim = task.get('min_grid_dim')
            max_grid_dim = task.get('max_grid_dim')
            
            # Generate 3 input-output examples for this task using generate_grid_examples
            valid = False
            while not valid:
                try:
                    examples = generate_grid_examples(
                        instructions, 
                        num_examples=3, 
                        grid_categories=grid_categories,
                        strict=False,  # Don't fail if we can't generate exactly 3
                        parameters=parameter_tags,
                        min_grid_dim=min_grid_dim,
                        max_grid_dim=max_grid_dim,
                        task_name=task.get('name'),
                        catch_exceptions=False
                    )
                    
                    if len(examples) < 3:
                        print(f"Warning: Only generated {len(examples)} examples for task {task_idx + 1}, skipping...")
                        continue
                    
                    # Use exactly 3 examples
                    examples = examples[:3]
                    
                    # Remove 'parameters' field from examples (not needed in validation.json format)
                    # and ensure 'object_mask' is present (it should be from generate_grid_examples now)
                    for example in examples:
                        if 'parameters' in example:
                            del example['parameters']
                        # Ensure object_mask exists (should be there from generate_grid_examples)
                        if 'object_mask' not in example:
                            example['object_mask'] = []
                    
                    # Create entry in the same format as validation.json
                    entry = {
                        'train': examples,
                        'test': [],  # No test examples for dreaming data
                        'prog': instructions,  # Instruction sequence (integer sequences)
                        'name': task.get('name', '')  # Task name
                    }
                    
                    samples.append(entry)
                    
                    valid = True

                except Exception as e:
                    continue

        return samples

    def _parse_task_programs(self, task_one, task_two):
        """Parse program strings for both tasks.
        
        Args:
            task_one: First task dictionary
            task_two: Second task dictionary
        
        Returns:
            Tuple of (prog_one_instrs, prog_two_instrs) or None if parsing fails
        """
        program_one_str = task_one.get('program', '')
        program_two_str = task_two.get('program', '')
        
        if not program_one_str or not program_two_str:
            return None
        
        # Parse programs to instruction strings
        prog_one_instrs = block_of_text_to_program_lines(program_one_str)
        prog_two_instrs = block_of_text_to_program_lines(program_two_str)
        
        return prog_one_instrs, prog_two_instrs

    def _check_get_instructions(self, prog_one_instrs, prog_two_instrs):
        """Check for get_objects and get_bg instructions in both tasks.
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
        
        Returns:
            Tuple of (has_get_objects_one, has_get_bg_one, has_get_objects_two, has_get_bg_two,
                     both_have_get_objects, both_have_get_bg)
        """
        has_get_objects_one = any('get_objects(' in instr for instr in prog_one_instrs)
        has_get_bg_one = any('get_bg(' in instr for instr in prog_one_instrs)
        has_get_objects_two = any('get_objects(' in instr for instr in prog_two_instrs)
        has_get_bg_two = any('get_bg(' in instr for instr in prog_two_instrs)
        
        both_have_get_objects = has_get_objects_one and has_get_objects_two
        both_have_get_bg = has_get_bg_one and has_get_bg_two
        
        return both_have_get_objects, both_have_get_bg

    def _remove_get_instructions_from_task_two(self, prog_two_instrs, both_have_get_objects, both_have_get_bg):
        """Remove get_objects and get_bg instructions from task_two.
        
        Args:
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions
        """
        new_prog_two_instrs = []
        
        for i, instr in enumerate(prog_two_instrs):
            should_remove = False
            if both_have_get_objects and instr.startswith('get_objects('):
                should_remove = True
            if both_have_get_bg and instr.startswith('get_bg('):
                should_remove = True
            
            if not should_remove:
                new_prog_two_instrs.append(instr)
        
        return new_prog_two_instrs

    def _remove_first_three_instructions_from_task_two(self, prog_two_instrs):
        """Remove the first three instructions (get_objects, get_bg, del(N+0)) from task_two.
        
        Args:
            prog_two_instrs: List of instruction strings from task_two
        
        Returns:
            Updated program instructions
        """
        if len(prog_two_instrs) < 3:
            return prog_two_instrs
        
        # Check if first three instructions match the expected pattern
        if (prog_two_instrs[0].startswith('get_objects(') and 
            prog_two_instrs[1].startswith('get_bg(') and 
            prog_two_instrs[2].startswith('del(') and 'N+0' in prog_two_instrs[2]):
            return prog_two_instrs[3:]
        
        return prog_two_instrs

    def _find_rebuild_grid_index(self, prog_instrs):
        """Find the index of the first rebuild_grid instruction.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Index of first rebuild_grid instruction, or None if not found
        """
        for i, instr in enumerate(prog_instrs):
            if 'rebuild_grid(' in instr:
                return i
        return None
    
    def _ends_with_crop(self, prog_instrs):
        """Check if a program ends with a crop instruction.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            True if the program has a crop instruction followed only by del statements (or nothing), False otherwise
        """
        if not prog_instrs:
            return False
        
        # Iterate backwards from the end, skipping del statements
        for i in range(len(prog_instrs) - 1, -1, -1):
            instr = prog_instrs[i]
            if instr.startswith('del('):
                # Skip del statements
                continue
            # Found a non-del instruction - check if it's a crop
            if instr.startswith('crop('):
                return True

            if instr.startswith('index('):  # TODO: this is not always technically correct, the index might not refer to a list of
                                            # objects and instead it might be a rebuild_grid task?
                return True

            if instr.startswith('rebuild_grid('):
                return False
        
        # All instructions were del statements (shouldn't happen, but handle it)
        return False
    
    def _ends_with_arg_min_or_max(self, prog_instrs):
        """Check if a program ends with an arg_min or arg_max instruction.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            True if the program has an arg_min or arg_max instruction followed only by del statements (or nothing), False otherwise
        """
        if not prog_instrs:
            return False
        
        # Iterate backwards from the end, skipping del statements
        for i in range(len(prog_instrs) - 1, -1, -1):
            instr = prog_instrs[i]
            if instr.startswith('del('):
                # Skip del statements
                continue
            # Found a non-del instruction - check if it's arg_min or arg_max
            if 'arg_min(' in instr or 'arg_max(' in instr:
                return True

            if instr.startswith('rebuild_grid('):
                return False
        
        # All instructions were del statements (shouldn't happen, but handle it)
        return False
    
    def _find_last_crop_index(self, prog_instrs):
        """Find the index of the last crop instruction.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Index of last crop instruction, or None if not found
        """
        last_crop_idx = None
        for i, instr in enumerate(prog_instrs):
            if 'crop(' in instr:
                last_crop_idx = i
        return last_crop_idx
    
    def _find_last_arg_min_or_max_index(self, prog_instrs):
        """Find the index of the last arg_min or arg_max instruction.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Index of last arg_min or arg_max instruction, or None if not found
        """
        last_idx = None
        for i, instr in enumerate(prog_instrs):
            if 'arg_min(' in instr or 'arg_max(' in instr:
                last_idx = i
        return last_idx
    
    def _swap_n0_to_n1_before_crop(self, prog_instrs):
        """Swap all N+0 to N+1 for all instructions up to (and excluding) the last crop.
        
        Args:
            prog_instrs: List of instruction strings
            comments: List of comments corresponding to instructions
        
        Returns:
            Tuple of (new_prog_instrs, new_comments)
        """
        crop_idx = self._find_last_crop_index(prog_instrs)
        if crop_idx is None:
            crop_idx = len(prog_instrs)
        
        new_prog_instrs = []
        
        for i, instr in enumerate(prog_instrs):
            if i < crop_idx:
                # Swap N+0 to N+1 for instructions before the last crop
                # Replace N+0 with N+1
                modified_instr = instr.replace('N+0', 'N+1')
                new_prog_instrs.append(modified_instr)
            else:
                # Keep instructions at and after the last crop unchanged
                new_prog_instrs.append(instr)
        
        return new_prog_instrs
    
    def _swap_n0_to_n1_before_arg_min_or_max(self, prog_instrs):
        """Swap all N+0 to N+1 for all instructions up to (and excluding) the last arg_min or arg_max.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Updated program instructions
        """
        arg_idx = self._find_last_arg_min_or_max_index(prog_instrs)
        if arg_idx is None:
            arg_idx = len(prog_instrs)
        
        new_prog_instrs = []
        
        for i, instr in enumerate(prog_instrs):
            if i < arg_idx:
                # Swap N+0 to N+1 for instructions before the last arg_min/arg_max
                # Replace N+0 with N+1
                modified_instr = instr.replace('N+0', 'N+1')
                new_prog_instrs.append(modified_instr)
            else:
                # Keep instructions at and after the last arg_min/arg_max unchanged
                new_prog_instrs.append(instr)
        
        return new_prog_instrs
    
    def _remove_instructions_from_rebuild_grid(self, prog_instrs):
        """Remove all instructions including and after the first rebuild_grid.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Updated program instructions
        """
        rebuild_idx = self._find_rebuild_grid_index(prog_instrs)
        if rebuild_idx is None:
            return prog_instrs
        
        # Remove rebuild_grid and everything after it
        return prog_instrs[:rebuild_idx]
    
    def _swap_n0_to_n1_before_rebuild_grid(self, prog_instrs):
        """Swap all N+0 to N+1 for all instructions up to (and excluding) rebuild_grid.
        
        Args:
            prog_instrs: List of instruction strings
        
        Returns:
            Updated program instructions
        """
        rebuild_idx = self._find_rebuild_grid_index(prog_instrs)
        if rebuild_idx is None:
            rebuild_idx = len(prog_instrs)
        
        new_prog_instrs = []
        
        for i, instr in enumerate(prog_instrs):
            if i < rebuild_idx:
                # Swap N+0 to N+1 for instructions before rebuild_grid
                # Replace N+0 with N+1
                modified_instr = instr.replace('N+0', 'N+1')
                new_prog_instrs.append(modified_instr)
            else:
                # Keep instructions at and after rebuild_grid unchanged
                new_prog_instrs.append(instr)
        
        return new_prog_instrs

    def composition_case1(self, prog_one_instrs, prog_two_instrs):
        """In this case, neither of the tasks are object tasks. Basic default case.
        Result: We simply concatenate the instruction lists.
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions for each of the two tasks (a tuple)
        """
        # Neither task is an object task, so no modifications needed - just concatenate
        return prog_one_instrs, prog_two_instrs

    def composition_case2(self, prog_one_instrs, prog_two_instrs):
        """task_one is an object task with rebuild_grid at the end, and task_two is also an
        object task with rebuild_grid at the end.
        Result: We remove from task_one the rebuild_grid instruction and everything after.
        From task_two, we remove the first 3 instructions and swap N+0 to N+1 until (and excluding)
        the last rebuild_grid.
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions for each of the two tasks (a tuple)
        """
        # Remove from task_one the rebuild_grid instruction and everything after
        prog_one_instrs = self._remove_instructions_from_rebuild_grid(
            prog_one_instrs
        )
        
        # Since task_two has rebuild_grid at the end, it must have both get_objects and get_bg
        # Remove its three starting instructions: get_objects, get_bg, and del(N+0)
        prog_two_instrs = self._remove_first_three_instructions_from_task_two(
            prog_two_instrs
        )
        
        # Swap all occurrences of N+0 to N+1 all the way to and excluding rebuild_grid
        prog_two_instrs = self._swap_n0_to_n1_before_rebuild_grid(
            prog_two_instrs
        )
        
        return prog_one_instrs, prog_two_instrs

    def composition_case3(self, prog_one_instrs, prog_two_instrs):

        """task_one is an object task with crop, arg_min, or arg_max at the end.
        task_two is a non-object task
        Result: We simply concatenate the instruction lists (like case 1)
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions for each of the two tasks (a tuple)
        """   
        # Similar to case 1 - no special handling for crop, just concatenate
        return prog_one_instrs, prog_two_instrs
        
    def composition_case4(self, prog_one_instrs, prog_two_instrs,
                          both_have_get_objects):
        """task_one is an object task with rebuild_grid at the end, and task_two is an
        object task with crop, arg_min, or arg_max at the end.
        Result: We remove from task_one the rebuild_grid instruction and everything after.
        From task_two, we remove the first 3 instructions and swap N+0 to N+1 until (and excluding)
        the last crop, arg_min, or arg_max instruction.
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions for each of the two tasks (a tuple)
        """
        # Remove from task_one the rebuild_grid instruction and everything after
        prog_one_instrs = self._remove_instructions_from_rebuild_grid(
            prog_one_instrs
        )
        
        # Check what get instructions task_two has
        has_get_objects_two = any('get_objects(' in instr for instr in prog_two_instrs)
        has_get_bg_two = any('get_bg(' in instr for instr in prog_two_instrs)
        
        # Check if task_two ends with crop or arg_min/arg_max
        ends_with_crop_two = self._ends_with_crop(prog_two_instrs)
        ends_with_arg_min_or_max_two = self._ends_with_arg_min_or_max(prog_two_instrs)
        
        # From task_two, remove the first 3 instructions and swap N+0 to N+1 until (and excluding) crop or arg_min/arg_max
        if has_get_objects_two and has_get_bg_two:
            # task_two uses both get_objects and get_bg:
            # Remove its three starting instructions: get_objects, get_bg, and del(N+0)
            prog_two_instrs = self._remove_first_three_instructions_from_task_two(
                prog_two_instrs
            )
            
            # Swap all occurrences of N+0 to N+1 all the way to and excluding the last crop or arg_min/arg_max
            if ends_with_crop_two:
                prog_two_instrs = self._swap_n0_to_n1_before_crop(
                    prog_two_instrs
                )
            elif ends_with_arg_min_or_max_two:
                prog_two_instrs = self._swap_n0_to_n1_before_arg_min_or_max(
                    prog_two_instrs
                )
        elif has_get_objects_two:
            # task_two uses only get_objects: delete the get_objects() instructions
            prog_two_instrs = self._remove_get_instructions_from_task_two(
                prog_two_instrs, both_have_get_objects, False
            )
            # Still need to swap N+0 to N+1 before crop or arg_min/arg_max
            if ends_with_crop_two:
                prog_two_instrs = self._swap_n0_to_n1_before_crop(
                    prog_two_instrs
                )
            elif ends_with_arg_min_or_max_two:
                prog_two_instrs = self._swap_n0_to_n1_before_arg_min_or_max(
                    prog_two_instrs
                )
        
        return prog_one_instrs, prog_two_instrs

    def _handle_get_instructions_special_case(self, prog_one_instrs, prog_two_instrs,
                                               both_have_get_objects, both_have_get_bg):
        """Handle the special case when both tasks have get_objects and/or get_bg.
        
        Args:
            prog_one_instrs: List of instruction strings from task_one
            prog_two_instrs: List of instruction strings from task_two
            both_have_get_objects: Whether both tasks have get_objects
            both_have_get_bg: Whether both tasks have get_bg
        
        Returns:
            Updated program instructions for each of the two tasks (a tuple)
        """
        # Determine which composition case we're in
        # Check if task_one is an object task (has get_objects)
        has_get_objects_one = any('get_objects(' in instr for instr in prog_one_instrs)
        
        # Check if task_one has rebuild_grid
        has_rebuild_one = self._find_rebuild_grid_index(prog_one_instrs) is not None
        
        # Check if task_one ends with crop (before any rebuild_grid removal)
        ends_with_crop_one = self._ends_with_crop(prog_one_instrs)
        # Check if task_one ends with arg_min or arg_max
        ends_with_arg_min_or_max_one = self._ends_with_arg_min_or_max(prog_one_instrs)
        
        # Check if task_two is an object task (has get_objects)
        has_get_objects_two = any('get_objects(' in instr for instr in prog_two_instrs)
        
        # Check if task_two has rebuild_grid
        has_rebuild_two = self._find_rebuild_grid_index(prog_two_instrs) is not None
        
        # Check if task_two ends with crop
        ends_with_crop_two = self._ends_with_crop(prog_two_instrs)
        # Check if task_two ends with arg_min or arg_max
        ends_with_arg_min_or_max_two = self._ends_with_arg_min_or_max(prog_two_instrs)
        
        # Determine composition case based on the comments in composition_case1/2/3/4 methods:
        # Case 1: Neither of the tasks are object tasks
        # Case 2: task_one is an object task with rebuild_grid at the end, and task_two is also an object task with rebuild_grid at the end
        # Case 3: task_one is an object task with crop, arg_min, or arg_max at the end
        # Case 4: task_one is an object task with rebuild_grid at the end, and task_two is an object task with crop, arg_min, or arg_max at the end
        #
        # Additionally, we want Case 3 to apply when task_one is an object task that:
        #   - has NO crop instruction anywhere
        #   - has NO rebuild_grid instruction
        #   - DOES contain arg_min or arg_max
        # In that situation we also just concatenate the instruction lists (same behavior as Case 3).
        
        # Helper flags for extended Case 3 condition
        has_any_crop_one = self._find_last_crop_index(prog_one_instrs) is not None
        has_arg_min_or_max_one = any(
            ('arg_min(' in instr) or ('arg_max(' in instr)
            for instr in prog_one_instrs
        )

        composition_case = None
        if not has_get_objects_two:
            composition_case = 1
        elif has_get_objects_one and has_rebuild_one and has_get_objects_two and has_rebuild_two:
            composition_case = 2
        elif has_get_objects_one and (
            ends_with_crop_one
            or ends_with_arg_min_or_max_one
            or (not has_any_crop_one and not has_rebuild_one and has_arg_min_or_max_one)
        ):
            composition_case = 3
        elif has_get_objects_one and has_rebuild_one and has_get_objects_two and (ends_with_crop_two or ends_with_arg_min_or_max_two):
            composition_case = 4
        
        # Validation: task_one is not a rebuild_grid task, task_two is an object task
        if (has_get_objects_one and not has_rebuild_one) and has_get_objects_two:
            #print("==> Invalid composition: task_one is not a rebuild_grid task and task_two is an object task")
            return None

        # Call the appropriate composition case method
        if composition_case == 1:
            return self.composition_case1(prog_one_instrs, prog_two_instrs)
        elif composition_case == 2:
            return self.composition_case2(prog_one_instrs, prog_two_instrs)
        elif composition_case == 3:
            return self.composition_case3(prog_one_instrs, prog_two_instrs)
        elif composition_case == 4:
            return self.composition_case4(prog_one_instrs, prog_two_instrs, both_have_get_objects)
        else:
            # Fallback: if no case matches, this is an error
            print(f"ERROR: Could not determine composition case!")
            print(f"  task_one has_get_objects: {has_get_objects_one}, has_rebuild: {has_rebuild_one}, ends_with_crop: {ends_with_crop_one}")
            print(f"  task_two has_get_objects: {has_get_objects_two}, has_rebuild: {has_rebuild_two}, ends_with_crop: {ends_with_crop_two}")
            print(f"  both_have_get_objects: {both_have_get_objects}, both_have_get_bg: {both_have_get_bg}")
            sys.exit(1)

    def _create_combined_task_dict(self, task_one, task_two, combined_program_str,
                                   combined_instructions):
        """Create the final combined task dictionary.
        
        Args:
            task_one: First task dictionary
            task_two: Second task dictionary
            combined_program_str: Combined program string
            combined_instructions: Combined instructions (token sequences)
            first_level: Curriculum level of task_one
            second_level: Curriculum level of task_two
        
        Returns:
            Combined task dictionary
        """
        # If both task names start with "Object", drop "Object" from the second one after the '+'
        _name1 = task_one.get('name', '') or ''
        _name2 = task_two.get('name', '') or ''
        if _name1.startswith("Object") and _name2.startswith("Object"):
            # Remove "Object" from start of second name (strip leading spaces or ':'/'-')
            trimmed_name2 = _name2[len("Object"):].lstrip(" :-")
            combined_name = f"{_name1} + {trimmed_name2}"
        else:
            combined_name = f"{_name1} + {_name2}"

        task_one_params = (task_one.get('parameters') or []) if isinstance(task_one.get('parameters'), list) else []
        task_two_params = (task_two.get('parameters') or []) if isinstance(task_two.get('parameters'), list) else []

        # Parameter tags from task_one prevail, otherwise use those from task_two
        # Result has max(len(task_one_params), len(task_two_params)) parameters
        max_len = max(len(task_one_params), len(task_two_params))
        param_tags = []
        for i in range(max_len):
            if i < len(task_one_params):
                param_tags.append(task_one_params[i])
            else:
                param_tags.append(task_two_params[i])
        return {
            'name': combined_name,
            'program': combined_program_str,
            'instructions': combined_instructions,
            'comments': [],
            'curriculum_level': task_one['curriculum_level'] + task_two['curriculum_level'] + 1,
            'source': 'Auto',
            'parameters': param_tags,
            'grid_categories': task_one.get('grid_categories', ['basic']),
            'min_grid_dim': max(task_one.get('min_grid_dim'), task_two.get('min_grid_dim')),
            'max_grid_dim': min(task_one.get('max_grid_dim'), task_two.get('max_grid_dim')),
            'validated': False
        }

    def combine_tasks(self, task_one, task_two):
        """Combine two tasks into a new combined task.
        
        Args:
            task_one: First task dictionary
            task_two: Second task dictionary
            first_level: Curriculum level of task_one
            second_level: Curriculum level of task_two
        
        Returns:
            Combined task dictionary, or None if combination fails
        """
        # Parse programs and prepare comments
        parsed = self._parse_task_programs(task_one, task_two)
        if parsed is None:
            return None
        prog_one_instrs, prog_two_instrs = parsed
        
        # Check for get_objects and get_bg instructions
        both_have_get_objects, both_have_get_bg = self._check_get_instructions(
            prog_one_instrs, prog_two_instrs
        )
        
        # Handle special case if both tasks have get_objects and/or get_bg
        result = self._handle_get_instructions_special_case(
            prog_one_instrs, prog_two_instrs,
            both_have_get_objects, both_have_get_bg
        )
        if result is None:
            return None
        prog_one_instrs, prog_two_instrs = result

        # Combine programs
        combined_prog_instrs = prog_one_instrs + prog_two_instrs
        
        # Reconstruct program string
        combined_program_str = program_lines_to_block_of_text(combined_prog_instrs)
        
        # Reconstruct instructions from the combined program
        combined_instructions = ProgUtils.convert_user_format_to_token_seq(combined_prog_instrs)
        if combined_instructions is None:
            print("==> ERROR: _reconstruct_instructions_from_program returned None!")
            return None
        
        # Create the combined task
        combined_task = self._create_combined_task_dict(
            task_one, task_two, combined_program_str, combined_instructions
        )
        
        return combined_task

    def _has_valid_object_mask(self, input_grid_np, object_mask):
        # Check if we have a valid object_mask
        # Tasks that require get_objects/get_bg need a valid object mask
        grid_height, grid_width = input_grid_np.shape[:2]
        
        # Check if object_mask is valid (not None, not empty list, not all zeros)
        has_valid_mask = False
        object_mask_np = None
        
        if object_mask is not None:
            # Convert object_mask to numpy if needed
            if isinstance(object_mask, list):
                # Empty list means no object mask
                if len(object_mask) == 0:
                    has_valid_mask = False
                else:
                    object_mask_np = np.array(object_mask, dtype=np.int32)
                    has_valid_mask = True
            else:
                object_mask_np = object_mask.copy()
                has_valid_mask = True
            
            # If we have an object_mask, ensure it's valid shape
            if has_valid_mask and object_mask_np is not None:
                # Ensure 2D
                if object_mask_np.ndim != 2:
                    if object_mask_np.size == grid_height * grid_width:
                        object_mask_np = object_mask_np.reshape(grid_height, grid_width)
                    else:
                        has_valid_mask = False
                elif object_mask_np.shape != (grid_height, grid_width):
                    if object_mask_np.size == grid_height * grid_width:
                        object_mask_np = object_mask_np.reshape(grid_height, grid_width)
                    else:
                        has_valid_mask = False
                
                # Check if mask is all zeros (empty mask - not valid)
                if has_valid_mask and np.all(object_mask_np == 0):
                    has_valid_mask = False

        return has_valid_mask, object_mask_np

    def _process_parameters(self, instructions_to_execute, parameter_tags, example_params, input_grid_np, attempt):
        # Handle parameters if needed
        # Only process parameters if the task actually has parameter tags
        # If parameter_tags is empty, skip parameter handling entirely (even if example_params is provided)
        if parameter_tags and len(parameter_tags) > 0:
            # Extract unique colors from input grid for parameter assignment
            unique_colors = np.unique(input_grid_np).tolist()
            if not unique_colors:
                unique_colors = [0]
            
            # Try to use parameters from example if available
            param_values = {}
            used_colors = set()
            
            # First, try to map example_params to our parameter indices
            if example_params:
                for param_name, param_value in example_params.items():
                    # param_name is like 'param1', 'param2', etc. (1-based)
                    # Convert to 0-based index
                    try:
                        param_idx = int(param_name.replace('param', '')) - 1
                        if 0 <= param_idx < len(parameter_tags):
                            param_values[param_idx] = param_value
                            if isinstance(param_value, (int, float)) and not isinstance(param_value, bool):
                                used_colors.add(int(param_value))
                    except (ValueError, AttributeError):
                        pass
            
            # Fill in remaining parameters
            for i, tag in enumerate(parameter_tags):
                if i in param_values:
                    # Already set from example_params
                    continue
                
                if tag == 'bg_color':
                    # Use a default bg_color
                    bg_color = 0 if 0 in unique_colors else (unique_colors[0] if unique_colors else 0)
                    param_values[i] = bg_color
                    used_colors.add(bg_color)
                elif tag in ('fg_color', 'color', 'existing_color'):
                    # Pick a color that hasn't been used
                    # Use deterministic selection based on attempt number
                    available_colors = sorted([c for c in unique_colors if c not in used_colors])
                    if available_colors:
                        # Use attempt number to cycle through available colors deterministically
                        color_idx = attempt % len(available_colors)
                        color = available_colors[color_idx]
                        param_values[i] = color
                        used_colors.add(color)
                    else:
                        param_values[i] = unique_colors[0] if unique_colors else 0
                elif tag == 'margin':
                    # Assign margin parameter (random integer between 1 and 5)
                    # Use attempt number to cycle through values deterministically
                    margin_value = (attempt % 5) + 1  # Values 1-5
                    param_values[i] = margin_value
                # For other tags, we can assign default values or skip
            
            if param_values:
                instructions_to_execute = replace_parameter_placeholders(
                    instructions_to_execute, param_values
                )
        
        return instructions_to_execute

    def _apply_task_to_input_grid(self, task, input_grid, object_mask=None, example_params=None, attempt=0):
        """Apply a task's program to a specific input grid.
        
        Args:
            task: Task dictionary with 'instructions' and 'parameters' fields
            input_grid: 2D list or numpy array representing the input grid
            object_mask: Optional object mask (2D list or numpy array)
            example_params: Optional dict of parameter values from example (e.g., {'param1': 5, 'param2': 3})
            attempt: Attempt number for trying different parameter combinations (0, 1, 2, ...)
        
        Returns:
            Output grid as numpy array, or None if execution fails
        """
        try:
            # Convert input grid to numpy if needed
            if isinstance(input_grid, list):
                input_grid_np = np.array(input_grid)
            else:
                input_grid_np = input_grid.copy()
            
            # Get task instructions and parameters
            instructions = task.get('instructions', [])
            if not instructions:
                return None
            
            parameter_tags = task.get('parameters', [])
            if not isinstance(parameter_tags, list):
                parameter_tags = []
            
            # Note: If parameter_tags is empty (like for "Object Flip V"), the parameter handling block
            # below will be completely skipped, and example_params (if provided) will be ignored.
            # This is correct behavior - tasks without parameters should execute without any parameter processing.
            
            # Convert to DSL GridObject
            input_grid_dsl = DSL.GridObject.from_grid(input_grid_np)
            initial_state = [[input_grid_dsl]]
            
            # Handle get_objects/get_bg if needed
            has_get_objects = len(instructions) > 0 and len(instructions[0]) > 1 and instructions[0][1] == 15
            has_get_bg = len(instructions) > 1 and len(instructions[1]) > 1 and instructions[1][1] == 16
            
            instructions_to_execute = instructions
            object_mask_np = None
            if has_get_objects or has_get_bg:
                has_valid_mask, object_mask_np = self._has_valid_object_mask(input_grid_np, object_mask)
                
                # If task requires object mask but we don't have a valid one, skip execution
                if not has_valid_mask:
                    return None
                
                # Extract objects and background
                grid_list, bg_grid = DSL.GridObject.get_grid_list([input_grid_dsl], [object_mask_np])
                
                if has_get_objects:
                    initial_state[0].append(grid_list[0])
                    if has_get_bg:
                        initial_state[0].append(bg_grid[0])
                        instructions_to_execute = instructions[2:]
                    else:
                        instructions_to_execute = instructions[1:]
                else:
                    initial_state[0].append(bg_grid[0])
                    instructions_to_execute = instructions[1:]
                        
            instructions_to_execute = self._process_parameters(instructions_to_execute, parameter_tags, example_params, input_grid_np, attempt)

            # Execute program
            # Convert object_mask to list format (pi.execute now expects a list of k object masks)
            # Use object_mask_np if available (processed version), otherwise use object_mask
            mask_to_use = object_mask_np if object_mask_np is not None else object_mask
            object_mask_list = [mask_to_use] if mask_to_use is not None else [None]
            debug_info = {}
            debug_info['task_name'] = task['name']
            output_grids_dsl = pi.execute(instructions_to_execute, initial_state, DSL, object_mask_list, debug_info)
            if output_grids_dsl and len(output_grids_dsl) > 0:
                output_grid_np = output_grids_dsl[0].cells_as_numpy()
                return output_grid_np
            
            return None
        except Exception as e:
            # Log the actual error for debugging (but don't print in normal operation to avoid spam)
            # The caller will handle the None return appropriately
            # Only print if it's not a common execution error (to reduce noise)
            # Uncomment the line below for detailed debugging:
            if DISPLAY_PROGRAM_VALIDATION_TRACEBACK:
                import traceback
                print(f"Error in _apply_task_to_input_grid for task '{task.get('name', 'unknown')}': {e}\n{traceback.format_exc()}")
            return None
    
    def _compare_output_grids(self, grid1, grid2):
        """Compare two output grids for equality.
        
        Args:
            grid1: First grid (2D list or numpy array)
            grid2: Second grid (2D list or numpy array)
        
        Returns:
            True if grids are equal, False otherwise
        """
        try:
            if isinstance(grid1, list):
                grid1 = np.array(grid1)
            if isinstance(grid2, list):
                grid2 = np.array(grid2)
            return np.array_equal(grid1, grid2)
        except Exception:
            return False
    
    def _predetermine_parameter_values(self, parameter_tags):
        """Randomly pre-determine parameter values for each param placeholder.
        
        Args:
            parameter_tags: List of parameter tags (e.g., ['bg_color', 'fg_color', 'margin'])
        
        Returns:
            Dict mapping parameter indices (0-based) to their predetermined values
        """
        if not parameter_tags or not isinstance(parameter_tags, list):
            return {}
        
        param_values = {}
        used_colors = set()
        
        # Pre-determine bg_color if needed (50% chance of 0, 50% chance of 1-9)
        bg_color = None
        for i, tag in enumerate(parameter_tags):
            if tag == 'bg_color':
                if random.uniform(0, 1) < 0.5:
                    bg_color = 0
                else:
                    bg_color = random.randint(1, 9)
                param_values[i] = bg_color
                used_colors.add(bg_color)
                break
        
        # Pre-determine color-related parameters
        for i, tag in enumerate(parameter_tags):
            if i in param_values:
                continue  # Already set (e.g., bg_color)
            
            if tag == 'fg_color':
                available_colors = [c for c in range(10) if c != bg_color and c not in used_colors]
                if not available_colors:
                    available_colors = [c for c in range(10) if c not in used_colors]
                if available_colors:
                    selected_color = random.choice(available_colors)
                    param_values[i] = selected_color
                    used_colors.add(selected_color)
            elif tag == 'color':
                available_colors = [c for c in range(10) if c not in used_colors]
                if available_colors:
                    selected_color = random.choice(available_colors)
                    param_values[i] = selected_color
                    used_colors.add(selected_color)
            elif tag == 'existing_color':
                # For pre-determination, pick a random color (will be validated against grid later)
                available_colors = [c for c in range(10) if c not in used_colors]
                if available_colors:
                    selected_color = random.choice(available_colors)
                    param_values[i] = selected_color
                    used_colors.add(selected_color)
            elif tag == 'margin':
                param_values[i] = random.randint(1, 6)  # Random margin value 1-5
        
        return param_values
    
    def _validate_mutation_produces_different_outputs(self, original_task, mutated_task):
        """Validate that mutated task produces different outputs than original task.
        
        Args:
            original_task: The original task dictionary
            mutated_task: The mutated task dictionary
        
        Returns:
            True if mutated task produces different outputs, False otherwise
        """
        # Generate examples from original task
        original_instructions = original_task.get('instructions', [])
        if not original_instructions:
            print("Original task has no instructions, skipping mutation validation.")
            return False
        
        original_grid_categories = original_task.get('grid_categories', ['basic'])
        original_parameter_tags = original_task.get('parameters', [])
        if not isinstance(original_parameter_tags, list):
            original_parameter_tags = []
        original_min_grid_dim = original_task.get('min_grid_dim')
        original_max_grid_dim = original_task.get('max_grid_dim')
        
        # Pre-determine parameter values for each param placeholder
        predetermined_param_values = self._predetermine_parameter_values(original_parameter_tags)
        
        # Get mutated task parameters (should be the same as original, but check to be safe)
        mutated_parameter_tags = mutated_task.get('parameters', [])
        if not isinstance(mutated_parameter_tags, list):
            mutated_parameter_tags = []
        
        try:
            # Generate examples from original task using predetermined parameter values
            grid_examples = generate_grid_examples(
                original_instructions,
                num_examples=10,
                grid_categories=original_grid_categories,
                strict=False,
                parameters=original_parameter_tags,
                min_grid_dim=original_min_grid_dim,
                max_grid_dim=original_max_grid_dim,
                parameter_values=predetermined_param_values,
                catch_exceptions=False,
                task_name = 'New mutation task'
            )
            
            if len(grid_examples) < 5:
                print("Not enough examples generated from original task, skipping mutation validation.")
                return False
        except:
            # If the original task doesn't run successfully, just pass validation and keep mutating.
            return True

        try:
            # Test on the examples
            num_test_examples = len(grid_examples)
            all_outputs_same = True
            
            for i in range(num_test_examples):
                original_example = grid_examples[i]
                
                # Use the same input grid from original example
                input_grid = original_example['input']
                object_mask = original_example.get('object_mask', [])
                example_params = original_example.get('parameters', {})
                
                # Apply original task
                original_output = self._apply_task_to_input_grid(
                    original_task, input_grid, object_mask, example_params=example_params, attempt=0
                )
                
                # Apply mutated task
                mutated_output = self._apply_task_to_input_grid(
                    mutated_task, input_grid, object_mask, example_params=example_params, attempt=0
                )
                
                if original_output is None and mutated_output is not None:
                    print("Mutation validation automatically passed: original task is invalid but mutated task is valid.")
                    return True
                
                if mutated_output is None:
                    return False
                
                # Check if outputs are the same for this example
                if not self._compare_output_grids(original_output, mutated_output):
                    all_outputs_same = False
            
            if all_outputs_same:
                if VERBOSE:
                    print("Mutation validation failed: mutated task produces same outputs as original task for all examples.")

                return False
            
            if VERBOSE:
                print("Mutation validation passed: mutated task produces different outputs.")
            return True
            
        except Exception as e:
            #print(f"Error during mutation validation: {e}")
            return False
    
    def _compare_two_object_masks(self, combined_object_masks, other_task):
        other_instructions = other_task.get('instructions', [])
        other_has_get_objects = len(other_instructions) > 0 and len(other_instructions[0]) > 1 and other_instructions[0][1] == 15
        other_has_get_bg = len(other_instructions) > 1 and len(other_instructions[1]) > 1 and other_instructions[1][1] == 16
        other_needs_object_mask = other_has_get_objects or other_has_get_bg
        
        # Check if combined task's examples have valid object masks
        # If other_task needs object masks but combined examples don't have them, skip this comparison
        if other_needs_object_mask:
            has_valid_masks = False
            for mask in combined_object_masks:
                if mask is not None:
                    if isinstance(mask, list):
                        if len(mask) > 0:
                            # Check if it's not all zeros
                            mask_np = np.array(mask)
                            if mask_np.size > 0 and not np.all(mask_np == 0):
                                has_valid_masks = True
                                break
                    else:
                        mask_np = np.array(mask)
                        if mask_np.size > 0 and not np.all(mask_np == 0):
                            has_valid_masks = True
                            break
            
            # If no valid object masks found, skip this task (can't apply object task to non-object grids)
            if not has_valid_masks:
                return False
        
        return True

    def _compare_two_tasks(self, task_one, other_task, examples, tasks_to_mark_obsolete, idx):
        '''
        Returns True if both tasks are the same
        '''
        combined_input_grids = [ex['input'] for ex in examples]
        combined_object_masks = [ex.get('object_mask', []) for ex in examples]
        combined_example_params = [ex.get('parameters', {}) for ex in examples]  # Store parameters from examples
        combined_output_grids = [ex['output'] for ex in examples]

        combined_level = task_one.get('curriculum_level', 0)        
        other_level = other_task.get('curriculum_level', 0)
        
        # Only check tasks with curriculum level <= combined task's level
        if other_level > combined_level:
            return False
        
        # Skip if this is the combined task itself (if it's already in the DB)
        if other_task.get('name') == task_one.get('name'):
            return False
        
        # Check if other_task requires object masks (has get_objects/get_bg)
        if not self._compare_two_object_masks(combined_object_masks, other_task):
            return False

        # Apply other task's program to the same input grids
        # Reuse the exact same parameter values that were used to generate the combined task's examples
        other_output_grids = []
        for i, input_grid in enumerate(combined_input_grids):
            object_mask = combined_object_masks[i] if i < len(combined_object_masks) else None
            # Use the exact parameters from the combined task example
            example_params = combined_example_params[i] if i < len(combined_example_params) else {}
            output_grid = self._apply_task_to_input_grid(
                other_task, input_grid, object_mask, example_params=example_params, attempt=0
            )
            if output_grid is not None:
                other_output_grids.append(output_grid)
            else:
                # If execution fails, we can't compare this task
                return False
        
        # Count how many outputs match
        matches = 0
        for combined_out, other_out in zip(combined_output_grids, other_output_grids):
            if self._compare_output_grids(combined_out, other_out):
                matches += 1

        match_percentage = matches / len(combined_output_grids) if combined_output_grids else 0

        # If more than N% match, handle according to program length
        if match_percentage > 0.9:
            # Calculate program length as total number of tokens in instruction sequences
            # Count the instructions based on the textual program ground truth, not the integer sequences
            program_text = task_one.get('program', '')
            combined_program_length = len([token for token in program_text.split(';') if token.strip() and not token.strip().startswith('del')])

            print(f"==> Redundancy validation failure: combined task {task_one['name']} clashed with {other_task['name']}")

            # Calculate program length as total number of tokens in instruction sequences
            other_program_text = other_task.get('program', '')
            other_program_length = len([token for token in other_program_text.split(';') if token.strip() and not token.strip().startswith('del')])
            
            if combined_program_length >= other_program_length:
                # Combined task is longer - fail validation
                return True
            else:
                # Other task is longer or equal - mark it as obsolete
                tasks_to_mark_obsolete.append(idx)
                return False

        return False

    def _validate_output_grids_distinct(self, combined_output_grids):
        # Validation: Check that at least 70% of output grids are distinct from each other
        unique_output_grids = []
        for output_grid in combined_output_grids:
            is_unique = True
            for unique_grid in unique_output_grids:
                if self._compare_output_grids(output_grid, unique_grid):
                    is_unique = False
                    break
            if is_unique:
                unique_output_grids.append(output_grid)
        
        distinct_percentage = len(unique_output_grids) / len(combined_output_grids) if combined_output_grids else 0
        if distinct_percentage < 0.7:
            print(f"==> Output variation validation failure: only {distinct_percentage*100:.1f}% of output grids are distinct (required: 70%)")
            return False

        return True

    def _validate_outputs_not_trivial(self, combined_output_grids):
        # Validation: Check that output grids are not all filled with exactly the same pixel color
        # Group examples into sets of 3 and check each set independently
        num_sets = len(combined_output_grids) // 3
        sets_with_uniform_same_color = 0
        
        for set_idx in range(num_sets):
            set_start = set_idx * 3
            set_output_grids = combined_output_grids[set_start:set_start + 3]
            
            # Check if all 3 grids in this set are uniform and have the same color
            first_grid = set_output_grids[0]
            first_color = None
            is_first_uniform = True
            
            # Check if first grid is uniform
            for row in first_grid:
                for pixel in row:
                    if first_color is None:
                        first_color = pixel
                    elif pixel != first_color:
                        is_first_uniform = False
                        break
                if not is_first_uniform:
                    break
            
            # If first grid is uniform, check if all grids in set have the same uniform color
            if is_first_uniform:
                all_same_color = True
                for output_grid in set_output_grids[1:]:
                    for row in output_grid:
                        for pixel in row:
                            if pixel != first_color:
                                all_same_color = False
                                break
                        if not all_same_color:
                            break
                    if not all_same_color:
                        break
                
                if all_same_color:
                    sets_with_uniform_same_color += 1
        
        # Fail if more than 70% of sets have all grids uniform with same color
        uniform_percentage = sets_with_uniform_same_color / num_sets if num_sets > 0 else 0
        if uniform_percentage > 0.7:
            print(f"==> Output validation failure: {uniform_percentage*100:.1f}% of example sets have all grids filled with the same pixel color (threshold: 70%)")
            return False

        return True

    def _validate_generated_task(self, combined_task, task_db):
        """Validate that combined task produces sufficiently different outputs from existing tasks.
        
        Args:
            combined_task: The newly combined task dictionary
            task_db: List of all tasks in the database
        
        Returns:
            Tuple of (is_valid, tasks_to_mark_obsolete)
            is_valid: True if task passes validation, False otherwise
            tasks_to_mark_obsolete: List of task indices that should be marked as [OBSOLETE]
        """
        try:
            # Check if combined_task's program already exists in task_DB
            combined_program = combined_task.get('program', '')
            if combined_program:
                for other_task in task_db:
                    other_program = other_task.get('program', '')
                    if other_program and other_program == combined_program:
                        # Exact program match found - fail validation
                        print(f"==> Program duplicate validation failure: combined task {combined_task.get('name', '')} has the same program as {other_task.get('name', '')}")
                        return False, []
            
            # Generate 25 examples from the combined task
            combined_instructions = combined_task.get('instructions', [])
            if not combined_instructions:
                print(f"==> combined instructions is empty -- why??")
                return False, []
            
            grid_categories = combined_task.get('grid_categories', ['basic'])
            parameter_tags = combined_task.get('parameters', [])
            if not isinstance(parameter_tags, list):
                parameter_tags = []
            min_grid_dim = combined_task.get('min_grid_dim')
            max_grid_dim = combined_task.get('max_grid_dim')
            
            # Generate examples
            examples = generate_grid_examples(
                combined_instructions,
                num_examples=25,
                grid_categories=grid_categories,
                strict=False,
                parameters=parameter_tags,
                min_grid_dim=min_grid_dim,
                max_grid_dim=max_grid_dim,
                catch_exceptions=False,
                task_name = f"New task: {combined_task['name']}"
            )

            if len(examples) < 25:
                # Not enough examples generated - fail validation
                print("Not enough examples generated -- failing validation")
                return False, []
            
            # Extract input grids and output grids from combined task examples
            combined_output_grids = [ex['output'] for ex in examples]
            
            if not self._validate_output_grids_distinct(combined_output_grids):
                return False, []

            if not self._validate_outputs_not_trivial(combined_output_grids):
                return False, []

            # Get combined task's curriculum level and program length           
            tasks_to_mark_obsolete = []
            
            # Loop through each other task in task_DB
            for idx, other_task in enumerate(task_db):
                is_same = self._compare_two_tasks(combined_task, other_task, examples, tasks_to_mark_obsolete, idx)
                if is_same:
                    return False, []

            # All checks passed
            return True, tasks_to_mark_obsolete
            
        except Exception as e:
            # If validation fails due to exception, fail validation
            if VERBOSE:
                print("==> An exception occurred: ")
                import traceback
                traceback.print_exc()

            return False, []

    def pick_two_tasks(self):
        # Get the curriculum level depths that exist right now in the task DB
        levels = set()
        for task in self.task_DB:
            lvl = task.get('curriculum_level')
            if isinstance(lvl, int):
                levels.add(lvl)
        
        # Pick two levels to combine, with a strong bias towards smaller values
        sorted_levels = sorted(levels)
        alpha = 3.0  # Controls sharpness of exponential bias

        # For each level, assign p_i proportional to exp(-alpha * i)
        probs = [math.exp(-alpha * i) for i in range(len(sorted_levels))]
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        # Pick two levels independently, allowing the same level to appear twice
        first_level = random.choices(sorted_levels, weights=probs, k=1)[0]
        second_level = random.choices(sorted_levels, weights=probs, k=1)[0]

        # Sample a random task for first_level
        first_level_tasks = [task for task in self.task_DB if task.get('curriculum_level') == first_level]
        second_level_tasks = [task for task in self.task_DB if task.get('curriculum_level') == second_level]
        
        # Filter out tasks with grid_category "croppable_corners" or "inside_croppable" from task_two selection
        second_level_tasks = [
            task for task in second_level_tasks
            if 'croppable_corners' not in task.get('grid_categories', []) and
            'inside_croppable' not in task.get('grid_categories', [])
        ]
        
        # Skip if no valid tasks remain for task_two after filtering
        if not second_level_tasks:
            print("No valid tasks for task_two after filtering (croppable_corners/inside_croppable excluded). Retrying...")
            return task_one, None

        task_one = random.choice(first_level_tasks)
        task_two = random.choice(second_level_tasks)

        return task_one, task_two

    def pick_one_task(self, other_than=None):
        possible_tasks = [task for task in self.task_DB if task != other_than] if other_than is not None else list(self.task_DB)
        if not possible_tasks:
            return None
        
        # 80% chance to prioritize tasks whose name contains "Object"
        if random.random() < 0.8:
            object_tasks = [task for task in possible_tasks if 'Object' in task.get('name', '')]
            if object_tasks:
                possible_tasks = object_tasks
        
        # Prioritize lower curriculum levels using exponential weighting
        alpha = 3.0  # Controls sharpness of exponential bias
        weights = [math.exp(-alpha * task.get('curriculum_level', 0)) for task in possible_tasks]
        return random.choices(possible_tasks, weights=weights, k=1)[0]

    def save_task(self, full_task_db, task_to_save, tasks_to_mark_obsolete):
        # Mark obsolete tasks and save everything in one operation
        task_db_path = Path("task_DB.json")
        
        # Mark obsolete tasks in full_task_db
        if tasks_to_mark_obsolete:
            for obsolete_idx in tasks_to_mark_obsolete:
                if obsolete_idx < len(full_task_db):
                    old_name = full_task_db[obsolete_idx].get('name', '')
                    if not old_name.startswith('[OBSOLETE]'):
                        full_task_db[obsolete_idx]['name'] = f'[OBSOLETE] {old_name}'
                        full_task_db[obsolete_idx]['validated'] = 'false'
        
        # Add the new combined task to full_task_db
        full_task_db.append(task_to_save)
        
        # Save everything at once (obsolete markers + new task)
        with open(task_db_path, "w") as f:
            json.dump(full_task_db, f, indent=2)
        
        # Update in-memory task_DB cache
        self.task_DB = full_task_db


    def apply_composition(self, task_one, task_two):

        #print(f"==> Combining {task_one['name']} and {task_two['name']}")

        # Validation: object task must be task_one, non-object task must be task_two
        # The reverse (non-object task_one + object task_two) is not allowed
        program_one_str = task_one.get('program', '')
        program_two_str = task_two.get('program', '')
        
        if program_one_str and program_two_str:
            prog_one_instrs = block_of_text_to_program_lines(program_one_str)
            prog_two_instrs = block_of_text_to_program_lines(program_two_str)
            
            has_get_objects_one = any('get_objects(' in instr for instr in prog_one_instrs)
            has_get_objects_two = any('get_objects(' in instr for instr in prog_two_instrs)
            
            # If task_two has get_objects but task_one doesn't, this is invalid
            # (object task must be task_one, non-object task must be task_two)
            if not has_get_objects_one and has_get_objects_two:
                #print("task mismatch: non-object task_one + object task_two combination not allowed. Validation failed.")
                return task_one, False

        # Combine the tasks
        combined_task = self.combine_tasks(task_one, task_two)

        return combined_task, True


    def _get_crossover_indices(self, prog_instrs, overlap_ok):
        """Get random from_idx and to_idx for crossover operation.
        
        Args:
            prog_instrs: List of program instructions
            overlap_ok: If True, allow from_idx == to_idx
        
        Returns:
            Tuple of (from_idx, to_idx) or None if task is invalid
        """
        program_length = len(prog_instrs)
        
        if program_length < 1:
            return None

        if overlap_ok:
            # allow from_idx == to_idx (so to_idx can be from 0 to program_length)
            to_idx = random.randint(0, program_length)
            from_idx = random.randint(0, to_idx)
        else:
            # from_idx < to_idx (to_idx at least 1, so from_idx in [0, to_idx-1])
            to_idx = random.randint(1, program_length)
            from_idx = random.randint(0, to_idx - 1)
        
        return from_idx, to_idx

    def stitch_programs(self, prog_one_instrs, task_one_from_idx, task_one_to_idx, prog_two_instrs, task_two_from_idx, task_two_to_idx):
        """Create a program by replacing a section of task_one with a section from task_two.
        
        Args:
            task_one: First task dictionary
            task_one_from_idx: Start index (inclusive) of section to replace in task_one
            task_one_to_idx: End index (exclusive) of section to replace in task_one.
                If equal to task_one_from_idx, no instructions are removed (insertion only).
            task_two: Second task dictionary
            task_two_from_idx: Start index (inclusive) of section to extract from task_two
            task_two_to_idx: End index (exclusive) of section to extract from task_two
        
        Returns:
            Combined program string, or None if stitching fails
        """
        # Validate indices
        # Allow task_one_from_idx == task_one_to_idx (insertion without removal)
        if (task_one_from_idx < 0 or task_one_to_idx > len(prog_one_instrs) or 
            task_one_from_idx > task_one_to_idx):
            return None
        
        if (task_two_from_idx < 0 or task_two_to_idx > len(prog_two_instrs) or 
            task_two_from_idx >= task_two_to_idx):
            return None
        
        # Extract section from task_two
        task_two_section = prog_two_instrs[task_two_from_idx:task_two_to_idx]
        
        # Replace section in task_one with reindexed section from task_two
        combined_instrs = (
            prog_one_instrs[:task_one_from_idx] + 
            task_two_section + 
            prog_one_instrs[task_one_to_idx:]
        )
        
        return combined_instrs

    def apply_crossover(self, task_one, task_two):

        prog_one = block_of_text_to_program_lines(task_one['program'])
        prog_two = block_of_text_to_program_lines(task_two['program'])

        prog_one = DreamingUtils.map_refIDs_to_uuids(prog_one)
        prog_two = DreamingUtils.map_refIDs_to_uuids(prog_two)

        prog_one = DreamingUtils.remove_dels(prog_one)
        prog_two = DreamingUtils.remove_dels(prog_two)

        # From task_one, randomly determine a "masked" region of the program ground truth
        task_one_indices = self._get_crossover_indices(prog_one, overlap_ok=True)
        if task_one_indices is None:
            return None, False
        task_one_from_idx, task_one_to_idx = task_one_indices

        # From task_two, pick the section that will be used to fill in task_one
        task_two_indices = self._get_crossover_indices(prog_two, overlap_ok=False)
        if task_two_indices is None:
            return None, False
        task_two_from_idx, task_two_to_idx = task_two_indices

        crossover_prog = self.stitch_programs(prog_one, task_one_from_idx, task_one_to_idx, prog_two, task_two_from_idx, task_two_to_idx)

        # properly "re-assign" UUID references to objects that are no longer available in the crossover program.
        range_start = task_one_from_idx
        crossover_prog = DreamingUtils.reassign_invalid_uuids(crossover_prog, range_start)
        crossover_prog = DreamingUtils.remove_unused_instructions(crossover_prog)        
        crossover_prog = DreamingUtils.auto_add_dels(crossover_prog)
        crossover_prog = DreamingUtils.map_uuids_to_refIDs(crossover_prog)

        crossover_prog_txt = program_lines_to_block_of_text(crossover_prog)
        crossover_instructions = ProgUtils.convert_user_format_to_token_seq(crossover_prog)
        if crossover_instructions is None:
            return None, False
       
        # Create the crossover task dictionary (similar to _create_combined_task_dict)
        _name1 = task_one.get('name', '') or ''
        _name2 = task_two.get('name', '') or ''
        if _name1.startswith("Object") and _name2.startswith("Object"):
            trimmed_name2 = _name2[len("Object"):].lstrip(" :-")
            crossover_name = f"{_name1} + {trimmed_name2} [crossover]"
        else:
            crossover_name = f"{_name1} + {_name2} [crossover]"

        task_one_params = (task_one.get('parameters') or []) if isinstance(task_one.get('parameters'), list) else []
        task_two_params = (task_two.get('parameters') or []) if isinstance(task_two.get('parameters'), list) else []
        
        max_len = max(len(task_one_params), len(task_two_params))
        param_tags = []
        for i in range(max_len):
            if i < len(task_one_params):
                param_tags.append(task_one_params[i])
            else:
                param_tags.append(task_two_params[i])

        # Extract parameter references from crossover_prog and filter param_tags
        param_refs = set()
        for match in re.finditer(r'param(\d+)', crossover_prog_txt):
            param_num = int(match.group(1))
            param_index = param_num - 1  # Convert to 0-based index (param1 -> 0, param2 -> 1, etc.)
            param_refs.add(param_index)
        
        # Filter param_tags to only include parameters that are actually referenced
        param_tags = [param_tags[i] for i in range(len(param_tags)) if i in param_refs]

        crossover_task = {
            'name': crossover_name,
            'program': crossover_prog_txt,
            'instructions': crossover_instructions,
            'comments': [],
            'curriculum_level': max(task_one.get('curriculum_level', 0), task_two.get('curriculum_level', 0)) + 1,
            'source': 'Auto',
            'parameters': param_tags,
            'grid_categories': task_one.get('grid_categories', ['basic']),
            'min_grid_dim': max(task_one.get('min_grid_dim', 3) or 3, task_two.get('min_grid_dim', 3) or 3),
            'max_grid_dim': min(task_one.get('max_grid_dim', 30) or 30, task_two.get('max_grid_dim', 30) or 30),
            'validated': False
        }

        return crossover_task, True

    def _count_stack_at_line(self, instructions, line_idx):
        """Calculate stack size after executing instructions up to (but not including) line_idx."""
        stack_size = 1  # Initial stack has N+0
        for i in range(line_idx):
            if i < len(instructions):
                instr = instructions[i].strip()
                if instr.startswith('del('):
                    stack_size -= 1
                    if stack_size < 0:
                        stack_size = 0
                else:
                    stack_size += 1
        return stack_size

    def _get_non_del_primitives(self):
        """Get all DSL primitives except 'del', 'identity', constants (0-9), and object attributes (starting with '.')."""
        non_del_prims = []
        for prim_name in DSL.semantics.keys():
            if (prim_name != 'del' and 
                prim_name != 'identity' and 
                not prim_name.isdigit() and 
                not prim_name.startswith('.')):
                non_del_prims.append(prim_name)
        return non_del_prims

    def _get_primitive_signature(self, prim_name):
        """Get the signature (argument types) of a primitive using ProgUtils.get_prim_func_arg_types when possible."""
        if prim_name not in DSL.semantics:
            return None
        
        prim_func = DSL.semantics[prim_name]
        
        # Skip constants and lambdas without annotations
        if isinstance(prim_func, int) or not callable(prim_func):
            return None
        
        try:
            # Try to use ProgUtils.get_prim_func_arg_types
            if prim_name in DSL.prim_indices:
                prim_idx = DSL.prim_indices[prim_name]
                if prim_idx in DSL.arg_counts:
                    nargs = DSL.arg_counts[prim_idx]
                    arg_types = ProgUtils.get_prim_func_arg_types(prim_idx + ProgUtils.NUM_SPECIAL_TOKENS, nargs)
                    # Convert type strings to match expected format (e.g., 'int' -> 'int', 'GridObject' -> 'GridObject')
                    return arg_types
        except Exception:
            pass
        
        # Fall back to manual signature parsing
        try:
            sig = inspect.signature(prim_func)
            annotations = prim_func.__annotations__
            param_names = list(sig.parameters.keys())
            
            arg_types = []
            for param_name in param_names:
                if param_name in annotations:
                    arg_types.append(str(annotations[param_name]))
                else:
                    arg_types.append(None)
            
            return arg_types
        except Exception:
            return None

    def _generate_typed_arguments(self, instructions, line_idx, arg_types):
        """Generate arguments by randomly choosing between constants and refIDs.
        
        Args:
            instructions: List of instruction strings
            line_idx: Line index where we're generating arguments
            arg_types: List of argument type strings from signature (ignored, kept for compatibility)
        
        Returns:
            List of argument values:
            - int 0-9 for constants
            - int >= 10 for stack vars (refID)
            - tuple (refID, attr_name) for attribute references where refID >= 10 and attr_name starts with "."
        """
        stack_size = self._count_stack_at_line(instructions, line_idx)
        nargs = len(arg_types) if arg_types else 0
        
        # Get all primitives that start with "." (attributes)
        dot_attributes = [name for name in DSL.semantics.keys() if name.startswith('.')]
        
        args = []
        for _ in range(nargs):
            # Randomly decide between constant (0-9) and refID (N+X)
            if stack_size > 0 and random.random() < 0.7:  # 70% chance to use refID if available
                # Pick a random refID that fits within stack_size
                var_idx = random.randint(0, stack_size - 1)
                ref_id = var_idx + 10  # Convert to refID format (10 = N+0, 11 = N+1, etc.)
                
                # Randomly decide whether to use refID as-is or as an attribute reference
                if dot_attributes and random.random() < 0.5:  # 50% chance to use attribute
                    attr_name = random.choice(dot_attributes)
                    args.append((ref_id, attr_name))
                else:
                    args.append(ref_id)
            else:
                # Use a constant
                args.append(random.randint(0, 9))
        
        return args

    def _create_instruction_string(self, prim_name, arg_refs):
        """Create an instruction string from primitive name and argument references.
        
        Args:
            prim_name: Name of the primitive
            arg_refs: List of argument references. Each is:
                     - An integer 0-9 for constants (formatted as "0", "1", etc.)
                     - An integer >= 10 for stack variable references (formatted as N+X where X = ref - 10)
                     - A tuple (var_ref, attr_name) for attribute references (formatted as N+X.attr)
        """
        if not arg_refs:
            return f"{prim_name}()"
        
        def format_arg(ref):
            if isinstance(ref, tuple):
                var_ref, attr_name = ref
                var_idx = var_ref - 10
                return f"N+{var_idx}{attr_name}"
            elif 0 <= ref <= 9:
                return str(ref)  # Constant
            else:
                var_idx = ref - 10  # Convert back to variable index
                return f"N+{var_idx}"
        
        args_str = ", ".join([format_arg(ref) for ref in arg_refs])
        return f"{prim_name}({args_str})"

    def _find_compatible_primitive(self, instructions, line_idx, max_attempts=50):
        """Try to find a primitive and generate arguments for it.
        
        Returns:
            Tuple of (prim_name, arg_refs) where arg_refs is a list where:
            - 0-9 represents constants
            - >= 10 represents stack variable references (var_idx + 10, so 10 = N+0, 11 = N+1, etc.)
        """
        non_del_prims = self._get_non_del_primitives()
        random.shuffle(non_del_prims)
        
        for prim_name in non_del_prims[:max_attempts]:
            sig = self._get_primitive_signature(prim_name)
            if sig is None:
                continue
            
            # Generate arguments (simplified, no type checking)
            arg_refs = self._generate_typed_arguments(instructions, line_idx, sig)
            return prim_name, arg_refs
        
        return None, None

    def _parse_instruction(self, instr_str):
        """Parse an instruction string into primitive name and arguments.
        
        Returns:
            Tuple of (prim_name, args_list) where args_list contains:
            - For constants: integer 0-9
            - For stack variables: integer >= 10 (var_idx + 10)
            - For attribute references: tuple (var_ref, attr_name) where var_ref is >= 10
        """
        match = re.match(r'(\w+)\((.*)\)', instr_str.strip())
        if not match:
            return None, []
        
        prim_name = match.group(1)
        args_str = match.group(2).strip()
        
        if not args_str:
            return prim_name, []
        
        # Parse arguments
        args = []
        current_arg = ""
        paren_depth = 0
        
        for char in args_str:
            if char == '(':
                paren_depth += 1
                current_arg += char
            elif char == ')':
                paren_depth -= 1
                current_arg += char
            elif char == ',' and paren_depth == 0:
                if current_arg.strip():
                    args.append(self._parse_argument(current_arg.strip()))
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(self._parse_argument(current_arg.strip()))
        
        return prim_name, args

    def _parse_argument(self, arg_str):
        """Parse a single argument string.
        
        Returns:
            - Integer 0-9 for constants
            - Integer >= 10 (var_idx + 10) for stack variable references
            - Tuple (var_ref, attr_name) for attribute references where var_ref >= 10
        """
        arg_str = arg_str.strip()
        
        # Check if it's an attribute reference (e.g., "N+0.x")
        if '.' in arg_str:
            parts = arg_str.split('.', 1)
            obj_ref = parts[0].strip()
            attr_name = '.' + parts[1].strip()
            
            # Parse object reference
            obj_match = re.match(r'N\+(\d+)', obj_ref)
            if obj_match:
                var_idx = int(obj_match.group(1))
                return (var_idx + 10, attr_name)
            return None
        
        # Check if it's a stack variable reference (N+X)
        if arg_str.startswith('N+'):
            ref_match = re.match(r'N\+(\d+)', arg_str)
            if ref_match:
                var_idx = int(ref_match.group(1))
                return var_idx + 10
        
        # Otherwise, try to parse as constant
        try:
            const_val = int(arg_str)
            if 0 <= const_val <= 9:
                return const_val
        except ValueError:
            pass
        
        return None

    def _generate_new_argument_value(self, arg_type, stack_size):
        """Generate a new argument value by randomly choosing between constants and refIDs.
        
        Args:
            arg_type: The current argument value (int 0-9 for constant, >= 10 for var)
            stack_size: Current stack size
            max_attempts: Maximum attempts to find a different value (ignored)
            var_types: Dict mapping var_idx -> type_string (ignored, kept for compatibility)
        
        Returns:
            New argument value (int 0-9 for constant, >= 10 for refID)
        """
        # Handle tuple (attribute references) - convert to simple refID
        if isinstance(arg_type, tuple):
            var_ref, _ = arg_type
            arg_type = var_ref
        
        if isinstance(arg_type, int):
            if 0 <= arg_type <= 9:
                # Constant: pick a different constant
                available_consts = [i for i in range(10) if i != arg_type]
                if available_consts:
                    return random.choice(available_consts)
                return arg_type  # Can't change, return same
            else:
                # Stack variable reference: randomly pick a different value
                var_idx = arg_type - 10
                
                # Randomly decide between constant and refID
                if stack_size > 0 and random.random() < 0.7:  # 70% chance to use refID if available
                    # Pick a different variable if possible
                    if stack_size > 1:
                        available_vars = [i for i in range(stack_size) if i != var_idx]
                        if available_vars:
                            new_var_idx = random.choice(available_vars)
                            return new_var_idx + 10
                    # If can't pick different variable, return same
                    return arg_type
                else:
                    # Use a constant instead
                    return random.randint(0, 9)
        
        # Fallback: return a random constant or refID
        if stack_size > 0 and random.random() < 0.7:
            var_idx = random.randint(0, stack_size - 1)
            return var_idx + 10
        else:
            return random.randint(0, 9)

    def _create_instruction_string_from_parts(self, prim_name, args):
        """Create an instruction string from primitive name and parsed arguments.
        
        Args:
            prim_name: Name of the primitive
            args: List of arguments (same format as _parse_argument returns)
        """
        if not args:
            return f"{prim_name}()"
        
        def format_arg(arg):
            if isinstance(arg, tuple):
                var_ref, attr_name = arg
                var_idx = var_ref - 10
                return f"N+{var_idx}{attr_name}"
            elif isinstance(arg, int):
                if 0 <= arg <= 9:
                    return str(arg)
                else:
                    var_idx = arg - 10
                    return f"N+{var_idx}"
            return str(arg)
        
        args_str = ", ".join([format_arg(arg) for arg in args])
        return f"{prim_name}({args_str})"

    def _mutation_add(self, instructions, line_idx):
        # 3. Add operation
        max_retries = 1000
        prim_name = None
        arg_refs = None
        found_match = False
        
        for _ in range(max_retries):
            prim_name, arg_refs = self._find_compatible_primitive(instructions, line_idx)
            if prim_name is None:
                continue
            found_match = True
            break
                        
        if not found_match or prim_name is None or arg_refs is None:
            return None
        
        # Create the new instruction
        new_instr = self._create_instruction_string(prim_name, arg_refs)
        
        # Calculate stack size at insertion point (before new instruction)
        stack_at_insert = self._count_stack_at_line(instructions, line_idx)
        new_var_ref = stack_at_insert  # The variable created by the new instruction
        
        # Insert the new instruction
        mutated_instrs = instructions[:line_idx] + [new_instr] + instructions[line_idx:]
        
        # Find a random position later in the program to add del statement
        # Must be after line_idx + 2 (after the new instruction)
        if len(mutated_instrs) > line_idx + 2:
            del_line_start = line_idx + 2
            del_line_end = len(mutated_instrs)
            del_line_idx = random.randint(del_line_start, del_line_end)
            
            # Between the inserted mutation and its corresponding del statement,
            # all instructions that refer to refIDs that have been created after
            # the inserted mutation must increment refIDs by 1.
            # Variables created at positions >= line_idx in the original program
            # now have indices +1, so we need to increment refIDs >= stack_at_insert
            # in instructions between line_idx + 1 and del_line_idx
            for i in range(line_idx + 1, del_line_idx):
                if i < len(mutated_instrs):
                    instr = mutated_instrs[i]
                    
                    # Update all N+X references that point to variables created after insertion
                    def update_ref(match):
                        ref_id = int(match.group(1))
                        # If this reference points to a variable created at or after insertion, increment it
                        if ref_id >= stack_at_insert:
                            return f"N+{ref_id + 1}"
                        return match.group(0)
                    
                    mutated_instrs[i] = re.sub(r'N\+(\d+)', update_ref, instr)
            
            # Track how the variable index changes due to del operations
            # between the insertion point and the del statement insertion point
            adjusted_var_ref = new_var_ref
            for i in range(line_idx + 1, del_line_idx):
                if i < len(mutated_instrs):
                    instr = mutated_instrs[i].strip()
                    if instr.startswith('del('):
                        # Parse the del instruction to get the variable index being deleted
                        prim_name, args = self._parse_instruction(instr)
                        if prim_name == 'del' and len(args) > 0:
                            # Extract variable index from argument (args[0] is var_idx + 10)
                            deleted_var_idx = args[0] - 10
                            # If a variable with index < adjusted_var_ref is deleted,
                            # our variable's index decreases by 1
                            if deleted_var_idx < adjusted_var_ref:
                                adjusted_var_ref -= 1
            
            # Add del statement (reference the variable created by new instruction)
            del_instr = f"del(N+{adjusted_var_ref})"
            mutated_instrs.insert(del_line_idx, del_instr)

        return mutated_instrs

    def _mutation_edit(self, instructions, line_idx):

        # Edit operation: decide a slot to edit
        current_instr = instructions[line_idx]
        prim_name, args = self._parse_instruction(current_instr)

        # Decide slot: primitive itself or one of its arguments
        if len(args) == 0:
            # No arguments, can only edit the primitive
            slot_type = 'primitive'
        else:
            # Randomly choose: primitive or one of the arguments
            slot_type = random.choice(['primitive', 'argument'])
        
        if slot_type == 'argument':
            arg_slot_idx = random.randint(0, len(args) - 1)

        stack_size = self._count_stack_at_line(instructions, line_idx)
        if slot_type == 'primitive':
            # Replace the primitive with a different non-del one
            non_del_prims = self._get_non_del_primitives()
            # Remove current primitive from candidates
            available_prims = [p for p in non_del_prims if p != prim_name]
            
            new_prim_name = random.choice(available_prims)
            
            # Get signature for new primitive
            sig = self._get_primitive_signature(new_prim_name)
            
            # Generate new arguments that match the signature types
            new_args = self._generate_typed_arguments(instructions, line_idx, sig)
            
            # Create new instruction
            new_instr = self._create_instruction_string_from_parts(new_prim_name, new_args)
            mutated_instrs = instructions[:]
            mutated_instrs[line_idx] = new_instr
        
        else:  # slot_type == 'argument'
            # Edit a specific argument
            current_arg = args[arg_slot_idx]
            
            # Generate a new value for this argument (simplified, no type checking)
            new_arg_value = self._generate_new_argument_value(current_arg, stack_size)
            
            # Create new args list with the edited argument
            new_args = args[:]
            new_args[arg_slot_idx] = new_arg_value
            
            # Create new instruction with edited argument
            new_instr = self._create_instruction_string_from_parts(prim_name, new_args)
            mutated_instrs = instructions[:]
            mutated_instrs[line_idx] = new_instr

        return mutated_instrs


    def apply_mutation(self, task):
        """Apply mutation to a task.
        
        1. Pick a random instruction line
        2. Pick whether to do an add or edit operation
        3. For add operation:
           - Pick a non-del instruction to add from DSL primitives
           - Pick parameter values based on argument types
           - Add a matching del statement later in the program
        4. For edit operation:
           - Decide a "slot" to edit (primitive or argument)
           - If primitive: replace with a different non-del primitive and randomize arguments
           - If argument: pick a different value for that argument based on its type
        """
        program_str = task.get('program', '')
        if not program_str:
            return None, False
        
        # Parse program to instruction strings
        instructions = block_of_text_to_program_lines(program_str)
        if len(instructions) == 0:
            return None, False
        
        # 1. Pick a random instruction line
        # For edit operation, we need an existing instruction, so line_idx must be < len(instructions)
        # For add operation, we can insert at any position including the end
        operation = random.choice(['add', 'edit'])
        
        if operation == 'edit':
            # For edit, we need an existing instruction
            if len(instructions) == 0:
                return None, False
            # Filter out 'del' statements - never pick them for mutation
            non_del_indices = [i for i in range(len(instructions)) 
                              if not instructions[i].strip().startswith('del(')]
            if len(non_del_indices) == 0:
                # All instructions are 'del' statements, cannot mutate
                return None, False
            line_idx = random.choice(non_del_indices)
        else:
            # For add, we can insert anywhere
            line_idx = random.randint(0, len(instructions))

        if operation == 'add':
            mutated_instrs = self._mutation_add(instructions, line_idx)

        elif operation == 'edit':
            mutated_instrs = self._mutation_edit(instructions, line_idx)

        if mutated_instrs is None:
            return None, False

        # Reconstruct program string
        mutated_prog = program_lines_to_block_of_text(mutated_instrs)

        # Convert instruction strings directly to token sequences to ensure they match the mutations
        mutated_instructions = []
        for instr_str in mutated_instrs:
            token_seq = ProgUtils.convert_user_instruction_to_token_seq(instr_str)
            if token_seq is None:
                #print(f"Mutation failed: cannot convert instruction string to token sequence: {instr_str}")
                return None, False
            mutated_instructions.append(token_seq)

        # Validate reference IDs
        valid = ProgUtils.validate_ref_ids(mutated_instrs)
        if not valid:
            return None, False
                
        # Create mutated task dictionary
        mutated_task = {
            'name': f"{task.get('name', '')} [mutated]",
            'program': mutated_prog,
            'instructions': mutated_instructions,
            'comments': [],
            'curriculum_level': task.get('curriculum_level', 0),
            'source': 'Auto',
            'parameters': task.get('parameters', []),
            'grid_categories': task.get('grid_categories', ['basic']),
            'min_grid_dim': task.get('min_grid_dim'),
            'max_grid_dim': task.get('max_grid_dim'),
            'validated': False
        }
        
        return mutated_task, True


    def apply_and_validate_composition(self, task, task_two, return_only):
        if VERBOSE:
            print("\tApplying composition")

        original_task = copy.deepcopy(task)
        new_task, is_valid = self.apply_composition(task, task_two)

        if new_task is None:
            is_valid = False

        if is_valid:
            # Validate the combined task against existing tasks
            # Load full task_DB for validation
            task_db_path = Path("task_DB.json")
            try:
                with open(task_db_path, "r") as f:
                    full_task_db = json.load(f)
            except Exception:
                full_task_db = []

            # Run validation
            is_valid, tasks_to_mark_obsolete = self._validate_generated_task(new_task, full_task_db)

            if is_valid:
                if VERBOSE:
                    print("\tValidation succeeded!")
                if return_only:
                    return new_task, is_valid, True

                self.save_task(full_task_db, new_task, tasks_to_mark_obsolete)                    

                return new_task, is_valid, True
            else:
                if VERBOSE:
                    print("\tValidation failed in validate_generated_task")
        else:
            if VERBOSE:
                print("\tapply_composition failed.")

            if new_task is None:
                new_task = original_task     # In the case of complete failure, revert to previous task

        return new_task, is_valid, False

    def apply_and_validate_crossover(self, task, task_two, return_only):
        if VERBOSE:
            print("\tApplying crossover")

        original_task = copy.deepcopy(task)            
        new_task, is_valid = self.apply_crossover(task, task_two)

        if new_task is None:
            is_valid = False

        if is_valid and new_task is not None:
            # Validate the combined task against existing tasks
            # Load full task_DB for validation
            task_db_path = Path("task_DB.json")
            try:
                with open(task_db_path, "r") as f:
                    full_task_db = json.load(f)
            except Exception:
                full_task_db = []

            # Run validation
            is_valid, tasks_to_mark_obsolete = self._validate_generated_task(new_task, full_task_db)

            if is_valid:
                if VERBOSE:
                    print("\tValidation succeeded!")

                if return_only:
                    return new_task, is_valid, True

                self.save_task(full_task_db, new_task, tasks_to_mark_obsolete)

                return new_task, is_valid, True
            else:
                if VERBOSE:
                    print("\tValidation failed in validate_generated_task")
        else:
            if VERBOSE:
                print("\tapply_crossover failed.")
            if new_task is None:
                new_task = original_task     # In the case of complete failure, revert to previous task

        return new_task, is_valid, False

    def apply_and_validate_mutation(self, task, return_only):
        if VERBOSE:
            print("\tApplying mutation")

        original_task = copy.deepcopy(task)
        new_task, is_valid = self.apply_mutation(task)

        if new_task is None:
            is_valid = False
        
        if is_valid and new_task is not None:
            # First validate that mutated task produces different outputs than original task
            is_valid = self._validate_mutation_produces_different_outputs(task, new_task)

            if is_valid:                    
                # Validate the combined task against existing tasks
                # Load full task_DB for validation
                task_db_path = Path("task_DB.json")
                try:
                    with open(task_db_path, "r") as f:
                        full_task_db = json.load(f)
                except Exception:
                    full_task_db = []

                # Run validation
                is_valid, tasks_to_mark_obsolete = self._validate_generated_task(new_task, full_task_db)

                if is_valid:
                    if VERBOSE:
                        print("\tValidation succeeded!")
                    if return_only:
                        return new_task, is_valid, True

                    self.save_task(full_task_db, new_task, tasks_to_mark_obsolete)

                    return new_task, is_valid, True
                else:
                    if VERBOSE:
                        print("\tValidation failed in validate_generated_task")
            else:
                if VERBOSE:
                    print("==> Mutated task generates the same outputs.")
        else:
            if new_task is None:
                new_task = original_task     # In the case of complete failure, revert to previous task

        return new_task, is_valid, False

    def dream_iteration(self, probs, max_step_count, return_only):
        
        multi_step_counter = max_step_count
        
        task = self.pick_one_task()
        task_two = self.pick_one_task(other_than=task)

        print(f"Attempting to generate new task from {task['name']} and {task_two['name']}...")

        already_composed = False    # Prevent multiple compositions in one dreaming iteration
        while multi_step_counter > 0:
            a = np.random.uniform()

            if VERBOSE:
                print(f"Attempt #{(max_step_count - multi_step_counter) + 1}...")

            if a < probs[0] and not already_composed:
                if VERBOSE:
                    print("==> Applying composition: ")
                new_task, is_valid, is_final = self.apply_and_validate_composition(task, task_two, return_only)
                if VERBOSE:
                    print(f"==> Composition result: is_valid = {is_valid}, is_final = {is_final}")
                if is_valid:
                    already_composed = True

            elif a < probs[1]:
                if VERBOSE:
                    print("==> Applying crossover: ")
                new_task, is_valid, is_final = self.apply_and_validate_crossover(task, task_two, return_only)
                if VERBOSE:
                    print(f"==> Crossover result: is_valid = {is_valid}, is_final = {is_final}")

            else:
                if VERBOSE:
                    print("==> Applying mutation: ")
                new_task, is_valid, is_final = self.apply_and_validate_mutation(task, return_only)
                if VERBOSE:
                    print(f"==> Mutation result: is_valid = {is_valid}, is_final = {is_final}")

            if is_final:
                return new_task 

            task = new_task
            multi_step_counter -= 1

        return None


def format_json_with_compact_integer_lists(obj, indent=2, current_indent=0):
    """Recursively format JSON with integer lists on a single line."""
    indent_str = ' ' * current_indent
    next_indent_str = ' ' * (current_indent + indent)
    
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        items = []
        for key, value in obj.items():
            formatted_value = format_json_with_compact_integer_lists(value, indent, current_indent + indent)
            items.append(f'{next_indent_str}"{key}": {formatted_value}')
        return '{\n' + ',\n'.join(items) + '\n' + indent_str + '}'
    
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        
        # Check if this is an integer list (all elements are numbers, not bool)
        is_integer_list = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
        
        if is_integer_list:
            # Format on single line
            return '[' + ', '.join(str(x) for x in obj) + ']'
        
        # Check if this is a list of integer lists
        is_list_of_integer_lists = all(
            isinstance(x, list) and len(x) > 0 and 
            all(isinstance(y, (int, float)) and not isinstance(y, bool) for y in x)
            for x in obj
        )
        
        if is_list_of_integer_lists:
            # Format each nested list on a single line, but keep outer list multi-line
            items = []
            for item in obj:
                formatted_item = '[' + ', '.join(str(x) for x in item) + ']'
                items.append(f'{next_indent_str}{formatted_item}')
            return '[\n' + ',\n'.join(items) + '\n' + indent_str + ']'
        
        # Regular list formatting
        items = []
        for item in obj:
            formatted_item = format_json_with_compact_integer_lists(item, indent, current_indent + indent)
            items.append(f'{next_indent_str}{formatted_item}')
        return '[\n' + ',\n'.join(items) + '\n' + indent_str + ']'
    
    else:
        # Primitive types
        return json.dumps(obj)

