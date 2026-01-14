import re
import AmotizedDSL.DSL as DSL
import numpy as np
from AmotizedDSL.prog_utils import ProgUtils


class DreamingUtils:
        
    @staticmethod
    def collect_used_uuids(uuid_instrs, uuid_pattern):
        used_uuids = set()

        for instr in uuid_instrs:
            instr_stripped = instr.strip()
            if instr_stripped.startswith('del('):
                # For del instructions, the UUID in del() is being used
                uuid_matches = re.findall(uuid_pattern, instr_stripped, re.IGNORECASE)
                used_uuids.update(uuid_matches)
            else:
                # For non-del instructions, extract UUIDs from arguments (the part after '=')
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr_stripped)
                if match:
                    instruction_part = match.group(2).strip()
                    # Extract all UUIDs from arguments
                    uuid_matches = re.findall(uuid_pattern, instruction_part, re.IGNORECASE)
                    used_uuids.update(uuid_matches)
        
        return used_uuids

    @staticmethod
    def mark_last_non_del_used(uuid_instrs, used_uuids, uuid_pattern):
        # Find the index of the last non-del instruction
        last_non_del_idx = None
        for i in range(len(uuid_instrs) - 1, -1, -1):
            if not uuid_instrs[i].strip().startswith('del('):
                last_non_del_idx = i
                break       
        
        if last_non_del_idx is not None:
            last_instr = uuid_instrs[last_non_del_idx].strip()
            if not last_instr.startswith('del('):
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', last_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        used_uuids.add(output_uuid_match.group(0))

        return last_non_del_idx

    @staticmethod
    def remove_unused_instructions(uuid_instrs):
        """Remove all instructions whose generated UUID is not used at all in the program.
        
        Args:
            crossover_instrs: List of instruction strings in format 
                '<uuid> = <instruction>(arguments)' or 'del(<uuid>)'
        
        Returns:
            Modified list of instruction strings with unused instructions removed
        """
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Collect all UUIDs that are used (appear in arguments or del statements)
        used_uuids = DreamingUtils.collect_used_uuids(uuid_instrs, uuid_pattern)

        # If there's a last non-del instruction, mark its output UUID as used
        last_non_del_idx = DreamingUtils.mark_last_non_del_used(uuid_instrs, used_uuids, uuid_pattern)
        
        # Filter out instructions whose output UUID is never used
        result = []
        for i, instr in enumerate(uuid_instrs):
            instr_stripped = instr.strip()
            if instr_stripped.startswith('del('):
                # Keep all del instructions
                result.append(instr)
            else:
                # For non-del instructions, check if output UUID is used
                # Always keep the last non-del instruction
                if i == last_non_del_idx:
                    result.append(instr)
                else:
                    match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr_stripped)
                    if match:
                        output_uuid_str = match.group(1).strip()
                        output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                        if output_uuid_match:
                            output_uuid = output_uuid_match.group(0)
                            # Only keep if the output UUID is used somewhere
                            if output_uuid in used_uuids:
                                result.append(instr)
                        else:
                            # If no valid output UUID, keep the instruction
                            result.append(instr)
                    else:
                        # If format doesn't match, keep the instruction
                        result.append(instr)
        
        return result

    @staticmethod
    def get_replacement_uuid_with_attr(valid_uuids, uuid_to_primitive, available_attributes, preserve_attr=None):
        """Get a replacement UUID, potentially with an attribute appended if it's a GridObject.
        
        Args:
            preserve_attr: If provided, randomly decide whether to preserve, change, or remove this attribute
        
        Returns:
            Replacement UUID with optional attribute
        """
        replacement = np.random.choice(list(valid_uuids))
        
        # Check if the replacement UUID's primitive returns GridObject
        is_gridobject = False
        if replacement in uuid_to_primitive:
            prim_name = uuid_to_primitive[replacement]
            return_type = ProgUtils.static_infer_result_type(prim_name)
            is_gridobject = (return_type == ProgUtils.TYPE_GRIDOBJECT or return_type == ProgUtils.TYPE_LIST_GRIDOBJECT)
        
        # If there was an original attribute, randomly decide what to do with it
        if preserve_attr:
            if not is_gridobject or not available_attributes:
                # If replacement is not a GridObject or no attributes available, remove attribute
                return replacement
            
            # Randomly decide: preserve (1/3), change (1/3), or remove (1/3)
            rand_val = np.random.random()
            if rand_val < 0.33:
                # Preserve the original attribute
                return replacement + preserve_attr
            elif rand_val < 0.67:
                # Change to a random attribute
                attr = np.random.choice(available_attributes)
                return replacement + attr
            else:
                # Remove the attribute
                return replacement
        
        # No original attribute - randomly decide whether to add one
        if is_gridobject:
            # Randomly decide whether to append an attribute
            if np.random.random() < 0.5 and available_attributes:
                # Append a random attribute
                attr = np.random.choice(available_attributes)
                return replacement + attr
        return replacement

    @staticmethod
    def process_instr_invalid_uuid(instr, valid_uuids, uuid_pattern, uuid_to_primitive, available_attributes):
        
        def perform_replacement(instruction_part, uuid_matches):
            instr_copy = instruction_part
            for match in reversed(uuid_matches):
                uuid_val = match.group(0)
                match_end = match.end()
                
                # Check if there's an attribute after the UUID
                attr_match = re.match(attr_pattern, instruction_part[match_end:])
                attr_str = None
                if attr_match:
                    attr_str = attr_match.group(0)
                    match_end = match_end + len(attr_str)
                
                # Check if base UUID (without attribute) is valid
                base_uuid = uuid_val
                if base_uuid not in valid_uuids:
                    # Get replacement UUID, preserving attribute if it exists
                    replacement = DreamingUtils.get_replacement_uuid_with_attr(valid_uuids, uuid_to_primitive,
                                                                                available_attributes, preserve_attr=attr_str)
                    instr_copy = instr_copy[:match.start()] + replacement + instr_copy[match_end:]

            return instr_copy

        # format is '<output_uuid> = <instruction>(arguments)'
        match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr)
        if match:
            output_uuid_str = match.group(1).strip()
            instruction_part = match.group(2).strip()
            
            # Extract output UUID
            output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
            output_uuid = None
            if output_uuid_match:
                output_uuid = output_uuid_match.group(0)
            
            # Extract primitive name for this instruction
            prim_match = re.match(r'^(\w+)\(', instruction_part)
            prim_name = prim_match.group(1) if prim_match else None
            
            # Extract all UUIDs from arguments (the instruction part)
            uuid_matches = list(re.finditer(uuid_pattern, instruction_part, re.IGNORECASE))
            
            # Pattern to match attributes (one or more attribute chains like .x, .y.x, etc.)
            attr_pattern = r'(\.\w+)+'
            
            # Replace invalid UUIDs in arguments
            # Note: 'input_grid' references won't match UUID pattern, so they won't be reassigned
            # (input_grid is always valid and is included in valid_uuids)
            instr_copy = perform_replacement(instruction_part, uuid_matches)

            # Reconstruct the instruction
            if output_uuid:
                # Add output UUID to valid set for subsequent instructions
                valid_uuids.add(output_uuid)
                # Store primitive name for this UUID
                if prim_name:
                    uuid_to_primitive[output_uuid] = prim_name

                return f"{output_uuid} = {instr_copy}"
            else:
                return f"{output_uuid_str} = {instr_copy}"
        else:
            # Fallback: if format doesn't match, try to replace UUIDs anyway
            uuid_matches = list(re.finditer(uuid_pattern, instr, re.IGNORECASE))
            attr_pattern = r'(\.\w+)+'
            return perform_replacement(instr, uuid_matches)            


    @staticmethod
    def reassign_invalid_uuids(uuid_no_del_instrs, range_start):
        """Reassign invalid UUIDs in a program in uuid-format with no del. (intermediate step for crossover operation)
        
        Goes through program instructions starting from range_start. For each instruction,
        if it has arguments that refer to a UUID that does not exist (i.e. there is no 
        previous instruction that generates that UUID), change it to a UUID that actually 
        exists (randomly selected among the previously generated UUIDs).
        
        When replacing a UUID, if the replacement UUID's primitive returns GridObject (type 1),
        randomly either leave the UUID as is, or append one of the possible attributes (like .x, .c, .max_y, etc.).
        
        Args:
            crossover_instrs: List of instruction strings in format 
                '<uuid> = <instruction>(arguments)' or 'del(<uuid>)'
            range_start: Index to start processing from
        
        Returns:
            Modified list of instruction strings with invalid UUIDs replaced
        """
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Collect all UUIDs generated by instructions before range_start and map them to their primitive names
        valid_uuids = set()
        uuid_to_primitive = {}  # Maps UUID to the primitive name that generated it
        
        # Add 'input_grid' as a valid absolute reference (always exists)
        valid_uuids.add('input_grid')
        
        for i in range(range_start):
            instr = uuid_no_del_instrs[i].strip()
            if not instr.startswith('del('):
                # Extract output UUID (the UUID before '=') and primitive name
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        output_uuid = output_uuid_match.group(0)
                        valid_uuids.add(output_uuid)
                        
                        # Extract primitive name from instruction part (e.g., "get_objects(...)" -> "get_objects")
                        instruction_part = match.group(2).strip()
                        prim_match = re.match(r'^(\w+)\(', instruction_part)
                        if prim_match:
                            uuid_to_primitive[output_uuid] = prim_match.group(1)
        
        # Get all available attributes from DSL (names starting with '.')
        available_attributes = [attr for attr in DSL.prim_indices.keys() if attr.startswith('.')]
        
        # If no valid UUIDs exist, we can't reassign anything
        if not valid_uuids:
            return uuid_no_del_instrs
                
        result = list(uuid_no_del_instrs)
        
        # Process instructions from range_start onwards
        for i in range(range_start, len(uuid_no_del_instrs)):
            instr = result[i].strip()
            
            result[i] = DreamingUtils.process_instr_invalid_uuid(instr, valid_uuids, uuid_pattern, uuid_to_primitive,
                                                                 available_attributes)
        
        return result

    @staticmethod
    def collect_output_uuids(uuid_instructions, uuid_pattern):
        # Collect all output UUIDs (the ones on the left side of '=')
        output_uuids = set()
        all_uuids = set()
        
        for instr in uuid_instructions:
            original_instr = instr.strip()
            if not original_instr.startswith('del('):
                # Non-del instruction: format is '<output_uuid> = <instruction>(arguments)'
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', original_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        output_uuids.add(output_uuid_match.group(0))
            
            # Collect all UUIDs from this instruction
            uuids = re.findall(uuid_pattern, original_instr, re.IGNORECASE)
            all_uuids.update(uuids)

        return output_uuids, all_uuids

    @staticmethod
    def map_uuid_to_refIDs_instruction(original_instr, uuid_pattern, stack):
        # Check if this is a del instruction or an assignment format instruction
        if original_instr.startswith('del('):
            # Delete instruction: format is 'del(<uuid>)'
            instr_copy = original_instr
            
            # Check if "input_grid" is being deleted (before replacement)
            has_input_grid = 'input_grid' in original_instr
            
            # Replace "input_grid" with "N+0" first
            instr_copy = instr_copy.replace('input_grid', 'N+0')
            
            # Extract all UUIDs from this instruction
            uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
            
            # Replace each UUID with its refID based on current stack state
            # Process in reverse order to maintain string positions
            for match in reversed(uuid_matches):
                obj_uuid = match.group(0)
                try:
                    idx = stack.index(obj_uuid)
                    ref_id = f"N+{idx}"
                    instr_copy = instr_copy[:match.start()] + ref_id + instr_copy[match.end():]
                except ValueError:
                    # UUID not in current stack - this shouldn't happen if the input is valid
                    # But handle gracefully by keeping the UUID
                    pass
            
            # Update stack: remove the deleted UUID
            if uuid_matches:
                del_uuid = uuid_matches[0].group(0)  # Get the UUID being deleted
                if del_uuid in stack:
                    stack.remove(del_uuid)
            elif has_input_grid:
                # Handle "input_grid" deletion - it was replaced with "N+0" but we need to remove it from stack
                if 'input_grid' in stack:
                    stack.remove('input_grid')

            return instr_copy
            
        else:
            # Non-del instruction: format is '<output_uuid> = <instruction>(arguments)'
            # Extract the output UUID (before the '=') and the instruction part (after the '=')
            match = re.match(r'^([^=]+)\s*=\s*(.+)$', original_instr)
            if match:
                output_uuid_str = match.group(1).strip()
                instruction_part = match.group(2).strip()
                
                # Extract the output UUID
                output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                output_uuid = output_uuid_match.group(0)
                
                # Replace each UUID in arguments with its refID based on current stack state
                instr_copy = instruction_part
                
                # First, handle "input_grid" references (they don't match UUID pattern)
                if 'input_grid' in instr_copy:
                    if 'input_grid' in stack:
                        idx = stack.index('input_grid')
                        ref_id = f"N+{idx}"
                        # Replace "input_grid" (and any attributes like "input_grid.x")
                        # Use regex to match "input_grid" followed by optional attribute
                        input_grid_pattern = r'input_grid(\.\w+)*'
                        instr_copy = re.sub(input_grid_pattern, lambda m: ref_id + (m.group(1) if m.group(1) else ''), instr_copy)
                
                # Then, extract and replace UUIDs
                uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                for match in reversed(uuid_matches):
                    obj_uuid = match.group(0)
                    try:
                        idx = stack.index(obj_uuid)
                        ref_id = f"N+{idx}"
                        # Check if there's an attribute after the UUID
                        attr_pattern = r'(\.\w+)+'
                        match_end = match.end()
                        attr_match = re.match(attr_pattern, instr_copy[match_end:])
                        attr_str = ''
                        if attr_match:
                            attr_str = attr_match.group(0)
                            match_end = match_end + len(attr_str)
                        instr_copy = instr_copy[:match.start()] + ref_id + attr_str + instr_copy[match_end:]
                    except ValueError:
                        # UUID not in current stack - this shouldn't happen if the input is valid
                        # But handle gracefully by keeping the UUID
                        pass
                
                # Update stack: add the output UUID
                stack.append(output_uuid)
                return instr_copy
            else:
                # Fallback: if format doesn't match, try old format
                instr_copy = original_instr
                # Replace "input_grid" with "N+0" first
                instr_copy = instr_copy.replace('input_grid', 'N+0')
                uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                for match in reversed(uuid_matches):
                    obj_uuid = match.group(0)
                    try:
                        idx = stack.index(obj_uuid)
                        ref_id = f"N+{idx}"
                        instr_copy = instr_copy[:match.start()] + ref_id + instr_copy[match.end():]
                    except ValueError:
                        pass

                return instr_copy


    @staticmethod
    def map_uuids_to_refIDs(uuid_instructions):
        """Transform absolute UUIDs back into relative refIDs (N+X) for each step.
        
        This is the reverse operation of map_refIDs_to_uuids.
        
        Args:
            uuid_instructions: List of instruction strings with UUIDs in format 
                ['<uuid> = <instruction>(arguments)', 'del(<uuid>)', ...]
        
        Returns:
            List of instruction strings with UUIDs replaced by refIDs relative to current stack size
        """
        # UUID pattern: 8-4-4-4-12 hexadecimal digits separated by hyphens
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        output_uuids, all_uuids = DreamingUtils.collect_output_uuids(uuid_instructions, uuid_pattern)

        if not all_uuids:
            return uuid_instructions
        
        # The initial UUID is the one that appears in arguments but is never an output
        # Find the first instruction and get the UUID from its arguments
        initial_uuid = None
        if uuid_instructions:
            first_instr = uuid_instructions[0].strip()
            if not first_instr.startswith('del('):
                # Extract UUIDs from arguments (the part after '=')
                match = re.match(r'^[^=]+\s*=\s*(.+)$', first_instr)
                if match:
                    instruction_part = match.group(1).strip()
                    arg_uuids = re.findall(uuid_pattern, instruction_part, re.IGNORECASE)
                    # The initial UUID is the one in arguments that is not an output
                    for uuid_val in arg_uuids:
                        if uuid_val not in output_uuids:
                            initial_uuid = uuid_val
                            break
        
        if initial_uuid is None:
            # Fallback: use first UUID that's not an output
            for uuid_val in all_uuids:
                if uuid_val not in output_uuids:
                    initial_uuid = uuid_val
                    break
        
        if initial_uuid is None:
            # Last resort: use first UUID encountered
            first_instr = uuid_instructions[0] if uuid_instructions else ""
            first_uuids = re.findall(uuid_pattern, first_instr, re.IGNORECASE)
            if first_uuids:
                initial_uuid = first_uuids[0]
        
        # Check if "input_grid" appears in any instruction - if so, it should be the initial object
        has_input_grid = any('input_grid' in instr for instr in uuid_instructions)
        
        # Now simulate the stack state at each instruction
        # Stack tracks which UUID is at each position
        # If input_grid is used, it should be the initial object (like in map_refIDs_to_uuids)
        if has_input_grid:
            stack = ["input_grid"]  # Initial object is "input_grid"
        elif initial_uuid is not None:
            stack = [initial_uuid]  # Initial object
        else:
            return uuid_instructions
        
        transformed = []
        
        for instr in uuid_instructions:

            original_instr = instr.strip()
            transformed_instr = DreamingUtils.map_uuid_to_refIDs_instruction(original_instr, uuid_pattern, stack)

            transformed.append(transformed_instr)

        return transformed

    @staticmethod
    def map_refIDs_to_uuids(prog_instructions):
        """Transform relative refIDs (N+X) into absolute UUIDs for each unique object.
        
        Args:
            prog_instructions: List of instruction strings like ["get_objects(N+0)", "del(N+0)", ...]
        
        Returns:
            List of instruction strings in format '<uuid> = <instruction>(arguments)' for non-del instructions,
            or 'del(<uuid>)' for del instructions, where refIDs are replaced by UUIDs
        """
        import uuid
        
        # Stack tracks which object UUID each position refers to
        # Stack starts with one object (N+0)
        stack = ["input_grid"]  # Initial object is "input_grid"
        
        # Map from refID (integer) to UUID for current stack state
        # N+0 refers to the first object (stack[0]), N+1 to second object (stack[1]), etc.
        def get_uuid_for_refid(ref_id):
            """Get the UUID for a refID based on current stack state."""
            if ref_id < len(stack):
                return stack[ref_id]
            return None
        
        transformed = []
        
        for instr in prog_instructions:
            original_instr = instr.strip()
            instr = original_instr
            
            # Extract all refIDs from this instruction
            ref_matches = list(re.finditer(r'N\+(\d+)', instr))
            
            # Replace each refID with its UUID
            # Process in reverse order to maintain string positions
            for match in reversed(ref_matches):
                ref_id = int(match.group(1))
                obj_uuid = get_uuid_for_refid(ref_id)
                if obj_uuid:
                    instr = instr[:match.start()] + obj_uuid + instr[match.end():]
            
            # Update stack based on instruction type (check original instruction)
            if original_instr.startswith('del('):
                # Delete instruction: extract refID and remove that object from stack
                del_match = re.search(r'del\(N\+(\d+)\)', original_instr)
                if del_match:
                    del_ref_id = int(del_match.group(1))
                    if del_ref_id < len(stack):
                        stack.pop(del_ref_id)
                # For del instructions, output format is just 'del(<uuid>)'
                transformed.append(instr)
            else:
                # Non-del instruction: creates new object, push to stack
                output_uuid = str(uuid.uuid4())
                stack.append(output_uuid)
                # Format as '<output_uuid> = <instruction>(arguments)'
                transformed.append(f"{output_uuid} = {instr}")
        
        return transformed

    @staticmethod
    def remove_dels(instructions):
        """Remove all del instructions from the instructions program.
        
        Args:
            instructions: List of instruction strings
            
        Returns:
            List of instruction strings with all del instructions removed
        """
        return [instr for instr in instructions if not instr.strip().startswith('del(')]

    @staticmethod
    def auto_add_dels(uuid_instructions):
        '''
        Here, those are instructions where the redIDs are actually UUIDs. They are absolute references.

        This function automatically adds a del statement once an object is no longer referenced for the rest of the
        program.
        '''
        # UUID pattern: 8-4-4-4-12 hexadecimal digits separated by hyphens
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Track all UUIDs that appear and which ones have been deleted
        all_uuids = set()
        deleted_uuids = set()
        
        # Track the last occurrence index of each UUID
        uuid_last_occurrence = {}
        
        # Track input_grid separately (it's not a UUID)
        input_grid_last_occurrence = None
        input_grid_deleted = False
        
        # First pass: find the last occurrence of each UUID and track deleted UUIDs
        for i, instr in enumerate(uuid_instructions):
            # Track deleted UUIDs
            if instr.strip().startswith('del('):
                deleted_uuids_in_instr = re.findall(uuid_pattern, instr, re.IGNORECASE)
                deleted_uuids.update(deleted_uuids_in_instr)
                # Check if input_grid is being deleted
                if 'input_grid' in instr:
                    input_grid_deleted = True
                continue
            
            # Extract all UUIDs from this instruction
            uuids = re.findall(uuid_pattern, instr, re.IGNORECASE)
            all_uuids.update(uuids)
            for uuid in uuids:
                uuid_last_occurrence[uuid] = i
            
            # Track input_grid occurrences
            if 'input_grid' in instr:
                input_grid_last_occurrence = i
        
        # Extract output UUID from the last instruction (if it exists)
        last_output_uuid = None
        if uuid_instructions:
            last_instr = uuid_instructions[-1].strip()
            if not last_instr.startswith('del('):
                # Extract output UUID (the UUID before '=')
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', last_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        last_output_uuid = output_uuid_match.group(0)
        
        # Second pass: insert del statements after last occurrences
        result = []
        dels_to_insert = {}  # Maps index -> list of UUIDs to delete at that position
        
        for uuid, last_idx in uuid_last_occurrence.items():
            # Insert del after the last instruction that references this UUID
            # Only if there are more instructions after it
            if last_idx < len(uuid_instructions) - 1:
                # Store at last_idx so we insert after that instruction
                if last_idx not in dels_to_insert:
                    dels_to_insert[last_idx] = []
                dels_to_insert[last_idx].append(uuid)
        
        # Handle input_grid deletion
        if input_grid_last_occurrence is not None and not input_grid_deleted:
            # Insert del(input_grid) after the last instruction that references it
            if input_grid_last_occurrence < len(uuid_instructions) - 1:
                if input_grid_last_occurrence not in dels_to_insert:
                    dels_to_insert[input_grid_last_occurrence] = []
                dels_to_insert[input_grid_last_occurrence].append('input_grid')
        
        # Build result with del statements inserted
        for i, instr in enumerate(uuid_instructions):
            result.append(instr)
            
            # Insert del statements after this instruction if needed
            if i in dels_to_insert:
                for uuid in dels_to_insert[i]:
                    result.append(f"del({uuid})")
                    if uuid != 'input_grid':
                        deleted_uuids.add(uuid)  # Track UUIDs deleted by inserted del statements
                    else:
                        input_grid_deleted = True  # Track that input_grid was deleted
        
        # Add del statements at the end for all non-deleted UUIDs except the last output
        remaining_uuids = all_uuids - deleted_uuids
        if last_output_uuid:
            remaining_uuids.discard(last_output_uuid)
        
        for uuid in sorted(remaining_uuids):
            result.append(f"del({uuid})")
        
        # Add del(input_grid) at the end if it hasn't been deleted already
        if input_grid_last_occurrence is not None and not input_grid_deleted:
            result.append("del(input_grid)")
        
        return result
