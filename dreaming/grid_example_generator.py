"""
Reusable module for generating input-output grid examples from program instructions.

This module encapsulates the logic for:
- Replacing parameter placeholders in instructions
- Generating input grids based on constraints
- Executing programs to produce output grids
- Handling special cases like get_objects/get_bg instructions
"""

import copy
import numpy as np
from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.DSL as DSL
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils


def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types (int64, float64, etc.)
    
    Returns:
        Object with all NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def find_max_parameter_number(instructions):
    """Find the maximum parameter number (param1, param2, etc.) in instructions.
    
    Args:
        instructions: The program instructions (may contain "param1", "param2", etc.)
    
    Returns:
        Maximum parameter number found (0 if no parameters)
    """
    max_param = 0
    
    def search_in_item(item):
        nonlocal max_param
        if isinstance(item, list):
            for subitem in item:
                search_in_item(subitem)
        elif isinstance(item, str) and item.startswith("param") and len(item) > 5:
            try:
                param_num = int(item[5:])
                if param_num > max_param:
                    max_param = param_num
            except ValueError:
                pass
    
    for instruction in instructions:
        search_in_item(instruction)
    
    return max_param

def replace_parameter_placeholders(instructions, parameters=None, parameter_values=None):
    """Replace parameter placeholders in instructions with actual values.
    
    Args:
        instructions: The program instructions (may contain "param1", "param2", etc.)
        example_index: Index of the example (0-based). param1 = example_index + 1, param2 = example_index + 2
        parameters: Optional list of tags, one per parameter index. Each tag applies to the parameter
                   at the corresponding index (param1 = index 0, param2 = index 1, etc.).
                   Tags can be: 'bg_color', 'fg_color', 'color', 'margin', 'existing_color', or empty string.
                   For backward compatibility, also accepts dict format with 'bg_color' or 'fg_color' keys.
        parameter_values: Optional dict mapping parameter indices (0-based) or param names (e.g., 'param1')
                         to their actual values. If provided, these values will be used instead of
                         example_index-based values.
    
    Returns:
        Instructions with parameter placeholders replaced by integer values
    """
    # Check if we need to modify instructions (i.e., if there are parameters to replace)
    # If no parameter_values provided, we still need to check if instructions contain param placeholders
    needs_copy = False
    if parameter_values and len(parameter_values) > 0:
        needs_copy = True
    else:
        # Quick check: see if instructions contain any param placeholders
        # Only do string conversion if we suspect there might be params
        def has_param_placeholders(obj):
            """Recursively check if object contains param placeholders."""
            if isinstance(obj, str):
                return obj.startswith("param") and len(obj) > 5 and obj[5:].isdigit()
            elif isinstance(obj, list):
                return any(has_param_placeholders(item) for item in obj)
            return False
        
        needs_copy = has_param_placeholders(instructions)
    
    if needs_copy:
        # Make a deep copy to avoid modifying the original
        instructions = copy.deepcopy(instructions)
    
    # Calculate parameter values       
    def replace_in_instruction(instruction):
        """Replace parameter placeholders in a single instruction sequence.
        
        Args:
            instruction: A single instruction sequence (list of tokens)
            is_first_in_sequence: Whether this is the first instruction in the sequence
        """
        if not instruction or not isinstance(instruction, list):
            return instruction
        
        result = []
        
        for item in instruction:
            if isinstance(item, list):
                result.append(replace_in_instruction(item))
            elif isinstance(item, str):
                # Replace parameter placeholders with actual values
                # IMPORTANT: Integer literals in token sequences must be offset by NUM_SPECIAL_TOKENS
                # When decoded, NUM_SPECIAL_TOKENS is subtracted, so we add it here to get the correct value
                if item.startswith("param") and len(item) > 5 and item[5:].isdigit():
                    # Extract parameter number (param1 -> 1, param2 -> 2, etc.)
                    param_num = int(item[5:])
                    param_index = param_num - 1  # Convert to 0-based index
                    
                    # Check if we have an explicit value for this parameter
                    replacement_value = None
                    if parameter_values:
                        # Try to get value by index first
                        if param_index in parameter_values:
                            replacement_value = parameter_values[param_index]
                        # Try to get value by param name
                        elif item in parameter_values:
                            replacement_value = parameter_values[item]
                    
                    # If no explicit value, determine based on tag or use default
                    if replacement_value is None:
                        # Get the tag for this parameter index
                        replacement_value = np.random.randint(0, 10)
                    
                    # Encode the value by adding NUM_SPECIAL_TOKENS
                    encoded_value = replacement_value + ProgUtils.NUM_SPECIAL_TOKENS
                    result.append(encoded_value)
                else:
                    result.append(item)
            else:
                result.append(item)
        
        # If the first token was a parameter placeholder that got replaced with a value,
        # we need to ensure SOS (0) is at the start of the instruction sequence.
        # Token sequences should always start with 0 (SOS), then primitive ID.
        # Check if we need to prepend SOS by looking at the original first token
        if len(instruction) > 0 and isinstance(instruction[0], str) and instruction[0].startswith("param"):
            # The first token was a parameter placeholder, so after replacement it's now a value
            # We need to prepend SOS (0) at the start
            if len(result) > 0 and result[0] != 0:
                result.insert(0, 0)
        
        return result
    
    # Replace parameters in each instruction
    replaced = []
    for i, instruction in enumerate(instructions):
        replaced.append(replace_in_instruction(instruction))
    
    return replaced

def assign_color_parameters(parameter_tags, unique_colors, bg_color, used_colors=None):
    """Assign colors to color-related parameters, ensuring mutual exclusivity.
    
    Args:
        parameter_tags: List of tags, one per parameter index (can be None, will default to empty list)
        unique_colors: List of unique colors present in the grid
        bg_color: Background color (can be None)
        used_colors: Optional set of already used colors (will be updated in place)
    
    Returns:
        Tuple of (param_values dict, updated used_colors set)
    """
    if used_colors is None:
        used_colors = set()
    
    # Handle None parameter_tags
    if parameter_tags is None:
        parameter_tags = []
    
    param_values = {}
    
    # Collect all color-related parameter indices
    color_param_indices = []
    for i, tag in enumerate(parameter_tags):
        if tag in ('color', 'fg_color', 'existing_color'):
            color_param_indices.append((i, tag))
    
    # Assign colors to color-related parameters, ensuring mutual exclusivity
    for i, tag in color_param_indices:
        if tag == 'existing_color':
            # Sample from unique colors in the grid, excluding already used colors
            available_colors = [c for c in unique_colors if c not in used_colors]
            if not available_colors:
                # If all unique colors are used, fall back to any unused color from 0-9
                available_colors = [c for c in range(10) if c not in used_colors]
            if available_colors:
                selected_color = np.random.choice(available_colors)
                param_values[i] = selected_color
                used_colors.add(selected_color)
        elif tag == 'fg_color':
            # Select a random color (0-9) excluding bg_color and already used colors
            available_colors = [c for c in range(10) if c != bg_color and c not in used_colors]
            if not available_colors:
                # If bg_color is the only option, we can't select fg_color
                # This shouldn't happen in practice, but handle it gracefully
                available_colors = [c for c in range(10) if c not in used_colors]
            if available_colors:
                selected_color = np.random.choice(available_colors)
                param_values[i] = selected_color
                used_colors.add(selected_color)
        elif tag == 'color':
            # Select a random color (0-9) excluding already used colors
            available_colors = [c for c in range(10) if c not in used_colors]
            if available_colors:
                selected_color = np.random.choice(available_colors)
                param_values[i] = selected_color
                used_colors.add(selected_color)
    
    # Handle bg_color parameter (if it's a parameter, not just for grid generation)
    for i, tag in enumerate(parameter_tags):
        if tag == 'bg_color' and bg_color is not None:
            param_values[i] = bg_color  # Use 0-based index
    
    return param_values, used_colors


def generate_grid(attributes, grid_categories, sampler=None):
    if sampler is None:
        sampler = GridSampler()

    # Use category-based sampling (similar to generators like triple_inversion_generator.py)
    # Build kwargs for sample_by_category
    sample_kwargs = {}
    sample_kwargs['bg_color'] = attributes['bg_color']
    sample_kwargs['min_dim'] = attributes['min_grid_dim']
    sample_kwargs['max_dim'] = attributes['max_grid_dim']
    
    # Try to pass all parameters, fall back if not supported
    try:
        input_grid, object_mask = sampler.sample_by_category(grid_categories, **sample_kwargs)

    except TypeError:
        # sample_by_category might not support some parameters
        # Fall back to sampling without optional parameters
        # Try with just grid_categories first
        try:
            print("TypeError occurred")
            input_grid, object_mask = sampler.sample_by_category(grid_categories)
        except:
            print("except")
            # If that fails, try with just bg_color if specified
            if attributes['bg_color'] is not None:
                input_grid, object_mask = sampler.sample_by_category(grid_categories, bg_color=attributes['bg_color'])
            else:
                print("raising exception")
                raise

    # Note: bg_color will still be used for parameter replacement in instructions
    input_grid_np = np.array(input_grid)

    return input_grid_np, object_mask


def generate_basic(attributes, parameter_tags, sampler=None, k=3, preset_parameter_values=None):
    if sampler is None:
        sampler = GridSampler()

    has_bg_color_param = parameter_tags and 'bg_color' in parameter_tags

    # Use min_grid_dim and max_grid_dim if provided
    sample_kwargs = {}
    sample_kwargs['min_dim'] = attributes['min_grid_dim']
    sample_kwargs['max_dim'] = attributes['max_grid_dim']

    if 'bg_color' in parameter_tags:
        if isinstance(parameter_tags, list):
            try:
                index = parameter_tags.index('bg_color')
                # Check preset values first
                if preset_parameter_values:
                    if index in preset_parameter_values:
                        sample_kwargs['bg_color'] = preset_parameter_values[index]
                    elif f'param{index + 1}' in preset_parameter_values:
                        sample_kwargs['bg_color'] = preset_parameter_values[f'param{index + 1}']
                # Fall back to attributes if not preset
                if 'bg_color' not in sample_kwargs:
                    param_key = f'param{index + 1}'
                    if param_key in attributes:
                        sample_kwargs['bg_color'] = attributes[param_key]
            except ValueError:
                pass  # 'bg_color' not found

    # If parameters include 'fg_color' or 'existing_color', preselect values and add to sample_kwargs['colors_present']
    colors_present = []
    if parameter_tags and isinstance(parameter_tags, list):
        for i, tag in enumerate(parameter_tags):
            if tag == 'fg_color':
                # Check preset values first
                fg_color = None
                if preset_parameter_values:
                    if i in preset_parameter_values:
                        fg_color = preset_parameter_values[i]
                    elif f'param{i + 1}' in preset_parameter_values:
                        fg_color = preset_parameter_values[f'param{i + 1}']
                
                # If not preset, randomly select
                if fg_color is None:
                    bg_color = sample_kwargs.get('bg_color', None)
                    possible_colors = [c for c in range(10) if c != bg_color]
                    if possible_colors:
                        fg_color = np.random.choice(possible_colors)
                    else:
                        fg_color = 0
                colors_present.append(fg_color)
            elif tag == 'existing_color':
                # Check preset values first
                c = None
                if preset_parameter_values:
                    if i in preset_parameter_values:
                        c = preset_parameter_values[i]
                    elif f'param{i + 1}' in preset_parameter_values:
                        c = preset_parameter_values[f'param{i + 1}']
                
                # If not preset, randomly select
                if c is None:
                    c = np.random.randint(0, 10)
                colors_present.append(c)
    if colors_present:
        sample_kwargs['colors_present'] = colors_present

    grids = []
    unique_colors_list = []

    for _ in range(k):
        input_grid = sampler.sample(**sample_kwargs)
        input_grid_np = np.array(input_grid)
        grids.append(input_grid_np)
        unique_colors = np.unique(input_grid_np).tolist()
        if not unique_colors:
            unique_colors = [0]
        unique_colors_list.append(unique_colors)

    # The main (first) grid for parameters assignment
    input_grid_np = grids[0]
    unique_colors = unique_colors_list[0]

    # Track used colors for parameter mutual exclusivity here
    used_colors = set()
    if has_bg_color_param and attributes['bg_color'] is not None:
        used_colors.add(attributes['bg_color'])

    # Start with preset values if provided
    param_values = {}
    if preset_parameter_values:
        # Convert string keys like 'param1' to integer indices
        for key, value in preset_parameter_values.items():
            if isinstance(key, str) and key.startswith('param') and len(key) > 5:
                try:
                    param_num = int(key[5:])
                    param_index = param_num - 1  # Convert to 0-based index
                    param_values[param_index] = value
                except ValueError:
                    pass
            elif isinstance(key, int):
                param_values[key] = value
    
    # Track colors already used by preset values
    if parameter_tags:
        for i, value in param_values.items():
            if i < len(parameter_tags) and parameter_tags[i] in ('color', 'fg_color', 'existing_color', 'bg_color'):
                used_colors.add(value)

    # Assign colors to color-related parameters using common function
    # Only assign if not already set by preset values
    generated_param_values, _ = assign_color_parameters(
        parameter_tags,
        unique_colors,
        attributes['bg_color'],
        used_colors
    )
    
    # Merge generated values (only for parameters not already set)
    for i, value in generated_param_values.items():
        if i not in param_values:
            param_values[i] = value
    
    # Assign margin parameters (random integer between 1 and 5)
    if parameter_tags:
        for i, tag in enumerate(parameter_tags):
            if tag == 'margin' and i not in param_values:
                param_values[i] = np.random.randint(1, 6)  # Random margin value 1-5

    # Return all 3 input grids, but for backward compatibility keep older return signature too
    return grids, param_values # No bg_mask for basic grids


def attempt_to_generate(placeholder_instructions, grid_categories, attributes, parameter_tags, sampler=None, preset_parameter_values=None, debug_info=None):
    # If bg_color or fg_color parameter is specified, randomly select bg_color for this example
    # 50% chance of being 0, 50% chance of being 1-9
    # Note: fg_color will be selected after grid generation to ensure mutual exclusivity
    
    has_bg_color_param = parameter_tags and 'bg_color' in parameter_tags
    has_fg_color_param = parameter_tags and 'fg_color' in parameter_tags

    bg_color = None
    # Check if bg_color is preset
    if preset_parameter_values:
        bg_color_index = None
        if parameter_tags:
            try:
                bg_color_index = parameter_tags.index('bg_color')
            except ValueError:
                pass
        
        if bg_color_index is not None:
            # Check preset values by index or param name
            if bg_color_index in preset_parameter_values:
                bg_color = preset_parameter_values[bg_color_index]
            elif f'param{bg_color_index + 1}' in preset_parameter_values:
                bg_color = preset_parameter_values[f'param{bg_color_index + 1}']
    
    # If not preset, randomly select
    if bg_color is None and (has_bg_color_param or has_fg_color_param):
        if np.random.uniform() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)
    
    attributes['bg_color'] = bg_color

    # Generate input grid based on constraints
    object_mask = []  # Initialize object_mask (will be empty for basic sampling)
    if grid_categories == ['basic']:
        grids, param_values = generate_basic(attributes, parameter_tags, sampler, preset_parameter_values=preset_parameter_values)
        
        # Use the first grid for execution
        input_grid_np = grids[0] if isinstance(grids, list) else grids

        # No need to skip instructions for basic tasks
        # Replace parameter placeholders with actual values for this example
        # replace_parameter_placeholders will handle copying if needed
        instructions_to_execute = replace_parameter_placeholders(placeholder_instructions, parameter_tags, param_values)

        # Convert to DSL GridObject
        input_grid_dsl = DSL.GridObject.from_grid(input_grid_np)
        initial_state = [[input_grid_dsl]]

        # Return all k=3 grids and empty object_masks list
        object_masks = [[] for _ in range(len(grids) if isinstance(grids, list) else 1)]
        return grids, object_masks, instructions_to_execute, initial_state, param_values
    else:
        # Check if instructions contain get_objects and/or get_bg anywhere in the program
        # In token sequence format: [SOS (0), primitive, SOP (1), args..., EOS (3)]
        # Primitive codes are offset by NUM_SPECIAL_TOKENS (4)
        # get_objects = 11 in DSL, so 11 + 4 = 15 in token sequence
        # get_bg = 12 in DSL, so 12 + 4 = 16 in token sequence
        has_get_objects = False
        has_get_bg = False
        get_objects_indices = []
        get_bg_indices = []
        
        for idx, instr in enumerate(placeholder_instructions):
            if len(instr) > 1:
                prim_code = instr[1]
                if prim_code == 15:  # get_objects
                    has_get_objects = True
                    get_objects_indices.append(idx)
                elif prim_code == 16:  # get_bg
                    has_get_bg = True
                    get_bg_indices.append(idx)
        
        # For object completion tasks (has_get_objects), use the same grid_category for all k examples
        categories_to_use = grid_categories
        if has_get_objects and len(grid_categories) > 1:
            # Select a single category randomly and use it for all k grids
            selected_category = np.random.choice(grid_categories)
            categories_to_use = [selected_category]
        
        # Generate k=3 input grids and object_masks
        k = 3
        grids = []
        object_masks = []

        for _ in range(k):
            input_grid_np, object_mask = generate_grid(attributes, categories_to_use, sampler)
            grids.append(input_grid_np)
            object_masks.append(object_mask)
        
            # Convert to DSL GridObject
            input_grid_dsl = DSL.GridObject.from_grid(input_grid_np)
            initial_state = [[input_grid_dsl]]

            # Only use object_mask if the program actually needs get_objects/get_bg
            if has_get_objects or has_get_bg:
                # Convert object_mask to numpy array and ensure it's 2D
                grid_height, grid_width = input_grid_np.shape[:2]
                
                if isinstance(object_mask, np.ndarray):
                    object_mask_np = object_mask.copy()
                else:
                    object_mask_np = np.array(object_mask, dtype=np.int32)
                
                # Ensure the mask is 2D and matches the grid shape
                if object_mask_np.ndim != 2:
                    if object_mask_np.size == grid_height * grid_width:
                        object_mask_np = object_mask_np.reshape(grid_height, grid_width)
                    else:
                        raise ValueError(f"object_mask size {object_mask_np.size} (shape {object_mask_np.shape}) doesn't match grid size {grid_height * grid_width}")
                elif object_mask_np.shape != (grid_height, grid_width):
                    if object_mask_np.size == grid_height * grid_width:
                        object_mask_np = object_mask_np.reshape(grid_height, grid_width)
                    else:
                        raise ValueError(f"object_mask shape {object_mask_np.shape} doesn't match grid shape ({grid_height}, {grid_width})")
                
                # Keep all instructions - get_objects and get_bg will be executed by the interpreter
                # with the object_mask and bg_mask passed to it
                instructions_copy = placeholder_instructions
            else:
                # No get_objects/get_bg needed - but we still want to keep object_mask for the output
                # object_mask is already available in this scope, we'll use it when creating the example
                instructions_copy = placeholder_instructions  # Will be copied in replace_parameter_placeholders if needed
        
        # Replace parameter placeholders with actual values for this example
        # Extract unique colors from the input grid for existing_color parameters
        # Cache unique_colors computation - only compute if needed
        unique_colors = None
        if parameter_tags and any(tag in ('existing_color', 'fg_color', 'color') for tag in parameter_tags):
            unique_colors = np.unique(input_grid_np).tolist()
            if not unique_colors:
                # Fallback if grid is empty (shouldn't happen, but be safe)
                unique_colors = [0]
        elif not unique_colors:
            unique_colors = []  # Empty list if not needed
        
        # Create parameter values dict based on tags and generated colors
        # Ensure all color-related parameters (color, fg_color, existing_color) are mutually exclusive
        param_values = {}
        
        # Start with preset values if provided
        if preset_parameter_values:
            # Convert string keys like 'param1' to integer indices
            for key, value in preset_parameter_values.items():
                if isinstance(key, str) and key.startswith('param') and len(key) > 5:
                    try:
                        param_num = int(key[5:])
                        param_index = param_num - 1  # Convert to 0-based index
                        param_values[param_index] = value
                    except ValueError:
                        pass
                elif isinstance(key, int):
                    param_values[key] = value
        
        if parameter_tags:
            # First, handle bg_color if present (it's used for grid generation, not as a parameter)
            # But we still need to track it for mutual exclusivity
            used_colors = set()
            if has_bg_color_param and bg_color is not None:
                used_colors.add(bg_color)
            
            # Track colors already used by preset values
            for i, value in param_values.items():
                if i < len(parameter_tags) and parameter_tags[i] in ('color', 'fg_color', 'existing_color', 'bg_color'):
                    used_colors.add(value)
            
            # Assign colors to color-related parameters using common function
            # Only assign if not already set by preset values
            generated_param_values, _ = assign_color_parameters(
                parameter_tags, 
                unique_colors, 
                bg_color, 
                used_colors
            )
            
            # Merge generated values (only for parameters not already set)
            for i, value in generated_param_values.items():
                if i not in param_values:
                    param_values[i] = value
            
            # Assign margin parameters (random integer between 1 and 5)
            for i, tag in enumerate(parameter_tags):
                if tag == 'margin' and i not in param_values:
                    param_values[i] = np.random.randint(1, 6)  # Random margin value 1-5
        
        # Pass the list of tags and the values dict
        instructions_to_execute = replace_parameter_placeholders(instructions_copy, parameter_tags, param_values)

        # Return all k=3 grids and object_masks
        # Return grids (list of k=3 grids) as input_grid_np, and object_masks (list of k=3 object_masks)
        # Also return bg_masks if they were created
        return grids, object_masks, instructions_to_execute, initial_state, param_values


def format_instruction_as_text(inst):
    """Convert a token sequence instruction to human-readable text.
    
    Args:
        inst: Token sequence instruction (list of integers)
    
    Returns:
        Human-readable string representation of the instruction
    """
    if not isinstance(inst, list) or len(inst) < 3:
        return str(inst)
    
    try:
        # Convert token sequence to intermediate representation (tuple)
        token_tuple = ProgUtils.convert_token_seq_to_token_tuple(inst, DSL)
        
        # Convert tuple to string representation
        prim_name, arg_strs = ProgUtils.convert_token_tuple_to_str(token_tuple, DSL)
        
        # Format arguments
        formatted_args = []
        for arg in arg_strs:
            if isinstance(arg, tuple):
                # Object-attribute pair
                formatted_args.append(f"({arg[0]}, {arg[1]})")
            else:
                formatted_args.append(str(arg))
        
        # Format as: primitive(arg1, arg2, ...)
        if formatted_args:
            return f"{prim_name}({', '.join(formatted_args)})"
        else:
            return prim_name
    except Exception as e:
        # Fallback to string representation if conversion fails
        return str(inst)


def execute_program(input_grid_np, instructions, initial_state, catch_exceptions=True, object_mask=None, debug_info=None):
    
    # Execute program
    if catch_exceptions:
        try:
            output_grids_dsl = pi.execute(instructions, initial_state, DSL, object_mask, debug_info)
            if output_grids_dsl and len(output_grids_dsl) > 0:
                output_grid_np = output_grids_dsl[0].cells_as_numpy()
                
                # Validation: If input grid is exactly identical to output grid, the example is invalid
                if np.array_equal(input_grid_np, output_grid_np):
                    # Skip this example and retry
                    return False, None, None
                
                if output_grid_np.shape[0] == 0 or len(output_grid_np.shape) != 2:
                    return False, None, None
                    
                # Validation: Output grid must be <= 30 wide and <= 30 in height
                output_height, output_width = output_grid_np.shape[:2]
                if output_width > 30 or output_height > 30:
                    # Skip this example and retry
                    return False, None, None
                
        except Exception as e:
            # If execution fails, capture the error information
            import traceback
            error_traceback = traceback.format_exc()
            print("Exception occurred during execute_program:")
            print(error_traceback)
            return False, None, error_traceback

        return True, output_grid_np, None
    else:
        output_grids_dsl = pi.execute(instructions, initial_state, DSL, object_mask, debug_info)
        if output_grids_dsl and len(output_grids_dsl) > 0:
            output_grid_np = output_grids_dsl[0].cells_as_numpy()

            # Validation: If input grid is exactly identical to output grid, the example is invalid
            if np.array_equal(input_grid_np, output_grid_np):
                # Skip this example and retry
                return False, None, None

            # Validation: Output grid must be <= 30 wide and <= 30 in height
            output_height, output_width = output_grid_np.shape[:2]
            if output_width > 30 or output_height > 30:
                # Skip this example and retry
                return False, None, None

        return True, output_grid_np, None


def detect_get_objects_get_bg(initial_state_template, instructions):
    """Detect if get_objects and/or get_bg are needed based on initial_state_template structure.
    
    Args:
        initial_state_template: The initial_state from attempt_to_generate
        instructions: Original instructions to check primitive IDs
    
    Returns:
        Tuple of (has_get_objects, has_get_bg)
    """
    has_get_objects = False
    has_get_bg = False
    
    if not (initial_state_template and len(initial_state_template) > 0 and len(initial_state_template[0]) > 1):
        return has_get_objects, has_get_bg
    
    state_len = len(initial_state_template[0])
    if state_len == 3:
        has_get_objects = True
        has_get_bg = True
    elif state_len == 2:
        # Check all instructions to find get_objects and/or get_bg
        # get_objects = 15, get_bg = 16 in token sequence
        for instr in instructions:
            if len(instr) > 1:
                prim_code = instr[1]
                if isinstance(prim_code, (int, np.integer)):
                    if prim_code == 15:  # get_objects
                        has_get_objects = True
                    elif prim_code == 16:  # get_bg
                        has_get_bg = True
        # Fallback: assume get_objects if we can't determine
        if not has_get_objects and not has_get_bg:
            has_get_objects = True
    
    return has_get_objects, has_get_bg


def create_initial_state_for_grid(input_grid_np):
    """Create initial_state for a single grid.
    
    Args:
        input_grid_np: Input grid as numpy array
        object_mask: Object mask (can be empty list, numpy array, or list) - not used here, passed to interpreter
        has_get_objects: Whether get_objects is needed - not used here, kept for compatibility
        has_get_bg: Whether get_bg is needed - not used here, kept for compatibility
    
    Returns:
        initial_state for this grid (just the input grid, get_objects/get_bg will be executed by interpreter)
    """
    input_grid_dsl = DSL.GridObject.from_grid(input_grid_np)
    initial_state = [[input_grid_dsl]]
    
    # No longer pre-populating state with get_objects/get_bg results
    # The interpreter will execute these instructions with the object_mask
    return initial_state


def execute_on_all_grids(input_grids_list, object_mask_list, instructions_to_execute, catch_exceptions=True, debug_info=None):
    """Execute program on all k input grids.
    
    Args:
        input_grids_list: List of k input grids
        object_mask_list: List of k object masks
        instructions_to_execute: Instructions to execute
        bg_mask: Optional background mask (single mask or list of masks, one per grid)
    
    Returns:
        Tuple of (all_valid, output_grids_list, error_traceback)
    """
    output_grids_list = []
    
    # Handle bg_mask: if it's a single mask, use it for all grids; if it's a list, use corresponding mask
    for idx, (input_grid_np, object_mask) in enumerate(zip(input_grids_list, object_mask_list)):
        initial_state = create_initial_state_for_grid(input_grid_np)
        
        valid_task, output_grid_np, error_traceback = execute_program(input_grid_np, instructions_to_execute, initial_state, catch_exceptions, object_mask, debug_info)
        
        if valid_task:
            output_grids_list.append(output_grid_np)
        else:
            return False, [], error_traceback
    
    return True, output_grids_list, None


def create_examples_from_grids(input_grids_list, output_grids_list, object_mask_list, param_values):
    """Create example dictionaries from input/output grids.
    
    Note: All k examples use the same parameter values (determined once from the first grid
    and baked into instructions_to_execute). Parameters are stored only in the first example
    to avoid duplication.
    
    Args:
        input_grids_list: List of k input grids
        output_grids_list: List of k output grids
        object_mask_list: List of k object masks
        param_values: Parameter values dict (same for all k examples)
    
    Returns:
        List of example dictionaries
    """
    # Convert param_values to parameters_dict
    parameters_dict = {}
    if param_values:
        for index, value in param_values.items():
            param_name = f'param{index + 1}'
            parameters_dict[param_name] = value
    
    examples = []
    for i, (input_grid_np, output_grid_np, object_mask) in enumerate(zip(input_grids_list, output_grids_list, object_mask_list)):
        example_dict = {
            'input': input_grid_np.tolist(),
            'output': output_grid_np.tolist(),
            'object_mask': object_mask if isinstance(object_mask, list) else (object_mask.tolist() if hasattr(object_mask, 'tolist') else object_mask),
        }
        # Only add parameters field if there are actual parameter values (add to first example only)
        # All k examples use the same parameter values, so we store them once to avoid duplication
        if parameters_dict:
            example_dict['parameters'] = parameters_dict
        examples.append(example_dict)
    
    return examples
    

def generate_grid_examples(instructions, num_examples=3, grid_categories=None, strict=True, parameters=None, min_grid_dim=None, max_grid_dim=None, 
                           parameter_values=None, catch_exceptions=True, task_name=None):
    """Generate input-output grid examples using the instructions.
    
    Args:
        instructions: The program instructions to execute (list of instruction sequences)
        num_examples: Number of examples to generate
        grid_categories: List of grid categories. If ['basic'] or None, uses sampler.sample().
                         Otherwise, uses sampler.sample_by_category() with the categories.
        strict: If True, raises ValueError if fewer than num_examples are generated.
                If False, returns whatever examples were generated (with a warning printed).
        parameters: Optional list of tags, one per parameter index. Each tag applies to the parameter
                   at the corresponding index (param1 = index 0, param2 = index 1, etc.).
                   Tags can be: 'bg_color', 'fg_color', 'color', 'margin', 'existing_color', or empty string.
                   For backward compatibility, also accepts dict format with 'bg_color' or 'fg_color' keys.
                   If 'bg_color' tag is present, bg_color will be randomly selected for each example
                   (50% chance of 0, 50% chance of 1-9) and used for grid generation.
                   If 'fg_color' tag is present, bg_color will be selected the same way for grid generation,
                   but param placeholders with 'fg_color' tag will be replaced with a random color (0-9) excluding bg_color.
                   If 'existing_color' tag is present, param placeholders will be replaced with a color that
                   exists in the input grid (randomly sampled from unique pixel values in the grid).
                   All color-related parameters ('color', 'fg_color', 'existing_color') are mutually exclusive:
                   no color value will be reused across different parameters of these types.
        min_grid_dim: Minimum grid dimension for sample_by_category calls. If None, not passed.
        max_grid_dim: Maximum grid dimension for sample_by_category calls. If None, not passed.
        parameter_values: Optional dict mapping parameter indices (0-based) or param names (e.g., 'param1')
                         to their actual values. If provided, these values will be used instead of
                         auto-generated values. Keys can be integers (0-based index) or strings like 'param1'.
                         If a parameter is not in this dict, it will be auto-generated as usual.
    
    Returns:
        List of dictionaries with 'input' and 'output' keys, each containing a 2D list of integers
        representing the grid cells.
    
    Raises:
        ValueError: If strict=True and fewer than num_examples are generated
    """
    if grid_categories is None:
        grid_categories = ['basic']

    examples = []
    max_attempts = num_examples * 10  # Try up to 10x the number of examples needed

    attempts = 0
    first_execution_error = None  # Store the first execution error encountered
    
    # Check if bg_color or fg_color parameter is needed
    # Handle new format: list of tags
    parameter_tags = []
    if parameters:
        if isinstance(parameters, list):
            parameter_tags = parameters
        elif isinstance(parameters, dict):
            # Backward compatibility: convert dict to list
            if parameters.get('bg_color') is not None:
                parameter_tags = ['bg_color']
            elif parameters.get('fg_color') is not None:
                parameter_tags = ['fg_color']
            else:
                parameter_tags = []
    
    # Reuse GridSampler instance across all attempts
    sampler = GridSampler()
    
    attributes = {
        'min_grid_dim': min_grid_dim,
        'max_grid_dim': max_grid_dim
    }
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        
        debug_info = None
        if task_name is not None:
            debug_info = {}
            debug_info['task_name'] = task_name

        input_grids_list, object_mask_list, instructions_to_execute, _, param_values = attempt_to_generate(
            instructions, grid_categories, attributes, parameter_tags, sampler, parameter_values)

        # Normalize to lists
        if not isinstance(input_grids_list, list):
            input_grids_list = [input_grids_list]
        if not isinstance(object_mask_list, list):
            object_mask_list = [object_mask_list]
        
        # Ensure object_mask_list has the same length as input_grids_list
        while len(object_mask_list) < len(input_grids_list):
            object_mask_list.append([])
        
        # Execute program on all k grids
        all_valid, output_grids_list, error_traceback = execute_on_all_grids(
            input_grids_list, object_mask_list, instructions_to_execute, catch_exceptions=catch_exceptions, debug_info=debug_info)
        
        # Store the first execution error we encounter
        if not all_valid and error_traceback and first_execution_error is None:
            first_execution_error = error_traceback
        
        # Create examples if all grids executed successfully
        if all_valid and len(output_grids_list) == len(input_grids_list):
            new_examples = create_examples_from_grids(input_grids_list, output_grids_list, object_mask_list, param_values)
            examples.extend(new_examples)

    if len(examples) < num_examples:
        error_msg = f"Only generated {len(examples)} out of {num_examples} requested examples after {max_attempts} attempts"
        if first_execution_error:
            error_msg += f"\n\nExecution error encountered:\n{first_execution_error}"
        if strict:
            raise ValueError(error_msg)
        else:
            import warnings
            warnings.warn(error_msg)

    # Convert all NumPy types to native Python types for JSON serialization
    examples_serializable = convert_numpy_types(examples)
    return examples_serializable

