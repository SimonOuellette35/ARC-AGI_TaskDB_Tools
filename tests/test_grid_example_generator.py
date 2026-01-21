from dreaming.dreaming_data_generator import DreamingDataGenerator, block_of_text_to_program_lines
from dreaming.grid_example_generator import generate_grid_examples, replace_parameter_placeholders
from AmotizedDSL.prog_utils import ProgUtils


def test_replace_parameter_placeholders():
    """Test replace_parameter_placeholders with various scenarios."""
    
    # Test 1: Basic replacement with single parameter by index
    prog1 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines1 = block_of_text_to_program_lines(prog1)
    instructions1 = ProgUtils.convert_user_format_to_token_seq(prog_lines1)
    
    parameter_values1 = {0: 5}  # param1 (index 0) = 5
    result1 = replace_parameter_placeholders(instructions1, parameter_values1)
    
    # Check that param1 was replaced with encoded value (5 + NUM_SPECIAL_TOKENS)
    assert len(result1) == 2
    assert "param1" not in str(result1)
    # Find the replaced value in the set_pixels instruction
    set_pixels_instr = result1[0]
    assert isinstance(set_pixels_instr, list)
    # The last argument before EOS (3) should be the encoded value
    encoded_value = set_pixels_instr[-2]  # Second to last (before EOS)
    assert encoded_value == 5 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 2: Replacement with parameter by name
    prog2 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines2 = block_of_text_to_program_lines(prog2)
    instructions2 = ProgUtils.convert_user_format_to_token_seq(prog_lines2)
    
    parameter_values2 = {"param1": 7}  # param1 = 7
    result2 = replace_parameter_placeholders(instructions2, parameter_values2)
    
    set_pixels_instr2 = result2[0]
    encoded_value2 = set_pixels_instr2[-2]
    assert encoded_value2 == 7 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 3: Multiple parameters
    prog3 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  set_pixels(N+0, 1, 1, \"param2\"),\n  del(N+0)\n]"
    prog_lines3 = block_of_text_to_program_lines(prog3)
    instructions3 = ProgUtils.convert_user_format_to_token_seq(prog_lines3)
    
    parameter_values3 = {0: 2, 1: 8}  # param1 = 2, param2 = 8
    result3 = replace_parameter_placeholders(instructions3, parameter_values3)
    
    # Check param1 replacement
    set_pixels_instr3a = result3[0]
    encoded_value3a = set_pixels_instr3a[-2]
    assert encoded_value3a == 2 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Check param2 replacement
    set_pixels_instr3b = result3[1]
    encoded_value3b = set_pixels_instr3b[-2]
    assert encoded_value3b == 8 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 4: Parameter without quotes (param1 instead of "param1")
    prog4 = "[\n  equal(N+0.c, param1),\n  del(N+0)\n]"
    prog_lines4 = block_of_text_to_program_lines(prog4)
    instructions4 = ProgUtils.convert_user_format_to_token_seq(prog_lines4)
    
    parameter_values4 = {0: 3}
    result4 = replace_parameter_placeholders(instructions4, parameter_values4)
    
    equal_instr = result4[0]
    # Find param1 position in equal instruction (should be after the property access)
    assert "param1" not in str(result4)
    # The param should be replaced with encoded value
    encoded_value4 = equal_instr[-2]  # Before EOS
    assert encoded_value4 == 3 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 5: Multiple parameters in same instruction
    prog5 = "[\n  switch(N+1, param1, param2),\n  del(N+1)\n]"
    prog_lines5 = block_of_text_to_program_lines(prog5)
    instructions5 = ProgUtils.convert_user_format_to_token_seq(prog_lines5)
    
    parameter_values5 = {0: 1, 1: 9}
    result5 = replace_parameter_placeholders(instructions5, parameter_values5)
    
    switch_instr = result5[0]
    assert "param1" not in str(result5)
    assert "param2" not in str(result5)
    # Both params should be replaced
    # Switch instruction format: [0, primitive, 1, args..., 3]
    # Find the replaced values
    param_positions = [i for i, val in enumerate(switch_instr) if isinstance(val, int) and val >= ProgUtils.NUM_SPECIAL_TOKENS and val < ProgUtils.NUM_SPECIAL_TOKENS + 10]
    assert len(param_positions) == 2
    replaced_values = [switch_instr[i] for i in param_positions]
    assert 1 + ProgUtils.NUM_SPECIAL_TOKENS in replaced_values
    assert 9 + ProgUtils.NUM_SPECIAL_TOKENS in replaced_values
    
    # Test 7: No parameter_values provided (should use random values)
    prog7 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines7 = block_of_text_to_program_lines(prog7)
    instructions7 = ProgUtils.convert_user_format_to_token_seq(prog_lines7)
    
    result7 = replace_parameter_placeholders(instructions7, None)
    
    # Should still replace param1 with some encoded value
    assert "param1" not in str(result7)
    set_pixels_instr7 = result7[0]
    encoded_value7 = set_pixels_instr7[-2]
    assert isinstance(encoded_value7, int)
    assert encoded_value7 >= ProgUtils.NUM_SPECIAL_TOKENS
    assert encoded_value7 < ProgUtils.NUM_SPECIAL_TOKENS + 10  # Random 0-9
    
    # Test 8: Instructions without parameters (should remain unchanged)
    prog8 = "[\n  set_pixels(N+0, 0, 0, 5),\n  del(N+0)\n]"
    prog_lines8 = block_of_text_to_program_lines(prog8)
    instructions8 = ProgUtils.convert_user_format_to_token_seq(prog_lines8)
    
    original8 = [instr[:] for instr in instructions8]  # Deep copy for comparison
    result8 = replace_parameter_placeholders(instructions8, {})
    
    # Should be identical (no params to replace)
    assert result8 == original8
    
    # Test 9: Complex nested structure with parameters
    prog9 = "[\n  add(N+0.x, 1),\n  equal(N+0.c, param1),\n  switch(N+1, param2, param1),\n  del(N+1)\n]"
    prog_lines9 = block_of_text_to_program_lines(prog9)
    instructions9 = ProgUtils.convert_user_format_to_token_seq(prog_lines9)
    
    parameter_values9 = {0: 6, 1: 2}
    result9 = replace_parameter_placeholders(instructions9, parameter_values9)
    
    # All params should be replaced
    assert "param1" not in str(result9)
    assert "param2" not in str(result9)
    
    # Check equal instruction
    equal_instr9 = result9[1]
    encoded_param1 = equal_instr9[-2]
    assert encoded_param1 == 6 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Check switch instruction
    switch_instr9 = result9[2]
    param_values_in_switch = [val for val in switch_instr9 if isinstance(val, int) and val >= ProgUtils.NUM_SPECIAL_TOKENS and val < ProgUtils.NUM_SPECIAL_TOKENS + 10]
    assert 2 + ProgUtils.NUM_SPECIAL_TOKENS in param_values_in_switch
    assert 6 + ProgUtils.NUM_SPECIAL_TOKENS in param_values_in_switch
    
    # Test 10: Empty parameter_values dict
    prog10 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines10 = block_of_text_to_program_lines(prog10)
    instructions10 = ProgUtils.convert_user_format_to_token_seq(prog_lines10)
    
    result10 = replace_parameter_placeholders(instructions10, {})
    
    # Should still replace with random value
    assert "param1" not in str(result10)
    set_pixels_instr10 = result10[0]
    encoded_value10 = set_pixels_instr10[-2]
    assert isinstance(encoded_value10, int)
    assert encoded_value10 >= ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 11: Parameter name takes precedence over index when both exist
    prog11 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines11 = block_of_text_to_program_lines(prog11)
    instructions11 = ProgUtils.convert_user_format_to_token_seq(prog_lines11)
    
    parameter_values11 = {"param1": 5}
    result11 = replace_parameter_placeholders(instructions11, parameter_values11)
    
    print(f"result11: {result11}")
    set_pixels_instr11 = result11[0]
    encoded_value11 = set_pixels_instr11[-2]
    # Should use param1 value (5), not index 0 value (1)
    assert encoded_value11 == 5 + ProgUtils.NUM_SPECIAL_TOKENS
    
    # Test 12: Original instructions should not be modified (deep copy)
    prog12 = "[\n  set_pixels(N+0, 0, 0, \"param1\"),\n  del(N+0)\n]"
    prog_lines12 = block_of_text_to_program_lines(prog12)
    instructions12 = ProgUtils.convert_user_format_to_token_seq(prog_lines12)
    original_instructions12 = [instr[:] for instr in instructions12]  # Deep copy
    
    parameter_values12 = {0: 4}
    result12 = replace_parameter_placeholders(instructions12, parameter_values12)
    
    # Original should still contain "param1"
    assert "param1" in str(original_instructions12)
    # Result should have replaced param1
    assert "param1" not in str(result12)
    # Original instructions should be unchanged
    assert instructions12 == original_instructions12

