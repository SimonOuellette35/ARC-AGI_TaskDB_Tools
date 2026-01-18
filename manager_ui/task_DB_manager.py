#!/usr/bin/env python3
"""
Simple HTTP server to serve the task manager webpage and handle JSON file operations.
Run this script and then open http://localhost:8000/manager_ui/task_manager.html in your browser.
"""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Path to task_DB.json in the root folder
TASK_DB_PATH = project_root / 'task_DB.json'

# Import required modules for grid generation
import AmotizedDSL.DSL as DSL
from AmotizedDSL.prog_utils import ProgUtils
from dreaming.grid_example_generator import generate_grid_examples
from dreaming.dreaming_data_generator import block_of_text_to_program_lines
import re


def count_parameters_in_program(program_str):
    """
    Count the number of parameters (param1, param2, etc.) in a program string.
    
    Args:
        program_str: Program string that may contain parameter placeholders
    
    Returns:
        Maximum parameter index found (0 if no parameters, 1 for param1, 2 for param2, etc.)
    """
    if not program_str or not isinstance(program_str, str):
        return 0
    
    max_param = 0

    # Find all parameter references like "param1", "param2", etc.
    matches = re.findall(r'param(\d+)', program_str)
    if matches:
        max_param = max(int(m) for m in matches)
    
    return max_param


def convert_parameters_to_tag_list(parameters, num_parameters):
    """
    Convert old parameter format to new list of tags format.
    
    Args:
        parameters: Old format - can be string, dict, or list
        num_parameters: Number of parameters expected (from counting param1, param2, etc. in program)
    
    Returns:
        List of tags, one per parameter index. Empty list if no parameters or invalid format.
    """
    if not parameters or num_parameters == 0:
        return []
    
    # If already a list, validate and return
    if isinstance(parameters, list):
        # Pad with empty strings if needed, or truncate if too long
        result = list(parameters[:num_parameters])
        while len(result) < num_parameters:
            result.append('')
        return result
    
    # Convert old format to new format
    tag = None
    if isinstance(parameters, str):
        tag = parameters if parameters else None
    elif isinstance(parameters, dict):
        # Handle dict format (e.g., {'bg_color': True} or {'type': 'bg_color'})
        if 'bg_color' in parameters or parameters.get('type') == 'bg_color' or parameters.get('parameter') == 'bg_color':
            tag = 'bg_color'
        elif 'fg_color' in parameters or parameters.get('type') == 'fg_color' or parameters.get('parameter') == 'fg_color':
            tag = 'fg_color'
        elif 'color' in parameters or parameters.get('type') == 'color':
            tag = 'color'
        elif 'margin' in parameters or parameters.get('type') == 'margin':
            tag = 'margin'
        elif 'existing_color' in parameters or parameters.get('type') == 'existing_color':
            tag = 'existing_color'
    
    # If we have a tag, apply it to all parameters
    if tag:
        return [tag] * num_parameters
    
    # Default: empty tags for all parameters
    return [''] * num_parameters


def normalize_program_format(program_str):
    """
    Normalize program string format to ensure it starts with "[\n" and ends with "\n]".
    
    Args:
        program_str: Program string that may or may not be properly formatted
    
    Returns:
        Normalized program string starting with "[\n" and ending with "\n]"
    """
    if not program_str or not isinstance(program_str, str):
        return program_str
    
    # Check if already properly formatted
    if program_str.startswith("[\n") and program_str.endswith("\n]"):
        return program_str
    
    # Extract the content (remove brackets and leading/trailing whitespace)
    content = program_str.strip()
    
    # Remove outer brackets if present
    if content.startswith('['):
        content = content[1:].lstrip()
    if content.endswith(']'):
        content = content[:-1].rstrip()
    
    # Remove leading/trailing newlines from content itself
    content = content.strip()
    
    # If content is empty, return minimal format
    if not content:
        return "[\n]"
    
    # Ensure it starts with "[\n" and ends with "\n]"
    # The content should preserve its internal structure (newlines, indentation, etc.)
    normalized = "[\n" + content + "\n]"
    
    return normalized


class TaskManagerHandler(BaseHTTPRequestHandler):
    def send_error_json(self, code, message):
        """Send an error response as JSON instead of HTML."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        error_response = json.dumps({
            'status': 'error',
            'message': message,
            'code': code
        })
        self.wfile.write(error_response.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests - serve HTML and JSON files."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        print(f"[DEBUG] GET request: path={path}, full_path={self.path}")

        # Serve task_manager.html
        if path == '/' or path == '/task_manager.html' or path == '/manager_ui/task_manager.html':
            self.serve_file('manager_ui/task_manager.html', 'text/html')
        # Serve task_DB.json
        elif path == '/task_DB.json' or path == '/manager_ui/task_DB.json':
            self.serve_file(str(TASK_DB_PATH), 'application/json')
        # Test endpoint
        elif path == '/test':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'message': 'Server is running'}).encode('utf-8'))
        # Generate grid examples
        elif path == '/generate_examples' or path.startswith('/generate_examples'):
            print(f"[DEBUG] Handling generate_examples request for path: {path}")
            self.handle_generate_examples()
        # Serve other static files
        else:
            # Check if it's a file that exists
            file_path = path.lstrip('/')
            if file_path and os.path.exists(file_path):
                self.serve_file(file_path, self.get_content_type(path))
            else:
                print(f"[DEBUG] 404: Path not found: {path} (file_path: {file_path})")
                # Check if it's an API endpoint (starts with / and doesn't look like a file)
                if path.startswith('/') and not path.endswith(('.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.json', '.ico')):
                    # Likely an API endpoint, return JSON error
                    self.send_error_json(404, f"Endpoint not found: {path}")
                else:
                    # Static file, return HTML error
                    self.send_error(404, "File not found")

    def do_POST(self):
        """Handle POST requests - save JSON data and execute programs."""
        if self.path == '/execute_program':
            self.handle_execute_program()
        elif self.path == '/save':
            try:
                # Read POST data
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    raise ValueError("No content provided")
                
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON data: {e}")
                
                # Validate data structure - should be a list
                if not isinstance(data, list):
                    raise ValueError(f"Expected a list of tasks, got {type(data).__name__}")
                
                # Validate each task has required fields (at minimum, name should be present)
                for i, task in enumerate(data):
                    if not isinstance(task, dict):
                        raise ValueError(f"Task at index {i} is not a dictionary")
                    # Name is required
                    if 'name' not in task or not task['name']:
                        raise ValueError(f"Task at index {i} is missing a name")
                
                # Normalize program format for tasks with program field
                for i, task in enumerate(data):
                    if 'program' in task and task.get('program'):
                        # Check if it's a new task (source is 'Manual' and no instructions yet)
                        is_new_task = (task.get('source') == 'Manual' and 
                                     ('instructions' not in task or not task.get('instructions')))
                        if is_new_task:
                            # Normalize program format to ensure it starts with "[\n" and ends with "\n]"
                            original_program = task['program']
                            normalized_program = normalize_program_format(original_program)
                            if normalized_program != original_program:
                                task['program'] = normalized_program
                                print(f"[SAVE] Normalized program format for task '{task.get('name', 'Unknown')}' at index {i}")
                
                # Generate instructions from program for tasks that have a program
                # Always regenerate instructions when program is present, even if instructions already exist
                N = len(DSL.semantics)
                for i, task in enumerate(data):
                    if 'program' in task and task['program']:
                        try:
                            program_str = task['program']
                            # Parse program string to hand-written format
                            program_hand_written = block_of_text_to_program_lines(program_str)
                            # Convert to instructions
                            instructions = ProgUtils.convert_user_format_to_token_seq(program_hand_written)
                            task['instructions'] = instructions
                            print(f"[SAVE] Regenerated instructions for task '{task.get('name', 'Unknown')}' at index {i}")
                        except Exception as e:
                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"[SAVE] Warning: Could not generate instructions for task '{task.get('name', 'Unknown')}' at index {i}: {e}")
                            print(f"[SAVE] Traceback: {error_trace}")
                            # Continue without instructions - task will be saved without them
                
                # Backup existing file before writing
                backup_path = str(TASK_DB_PATH) + '.backup'
                if os.path.exists(TASK_DB_PATH):
                    try:
                        with open(TASK_DB_PATH, 'r') as f:
                            backup_data = f.read()
                        with open(backup_path, 'w') as f:
                            f.write(backup_data)
                        print(f"[SAVE] Created backup: {backup_path}")
                    except Exception as e:
                        print(f"[SAVE] Warning: Could not create backup: {e}")
                
                # Write new data to task_DB.json
                with open(TASK_DB_PATH, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"[SAVE] Successfully saved {len(data)} task(s) to task_DB.json")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'success',
                    'message': f'Saved {len(data)} task(s) successfully',
                    'count': len(data)
                }).encode('utf-8'))
                
            except ValueError as e:
                # Client error (bad data)
                print(f"[SAVE] Error: {e}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }).encode('utf-8'))
            except Exception as e:
                # Server error
                import traceback
                error_trace = traceback.format_exc()
                print(f"[SAVE] Server error: {error_trace}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def handle_generate_examples(self):
        """Handle request to generate grid examples for a task."""
        try:
            # Parse query parameters
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            try:
                task_index = int(query_params.get('task_index', [0])[0])
                num_examples = int(query_params.get('num_examples', [3])[0])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid query parameters: {e}")

            # Load task_DB.json
            try:
                with open(TASK_DB_PATH, 'r') as f:
                    tasks = json.load(f)
            except FileNotFoundError:
                raise ValueError("task_DB.json not found")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in task_DB.json: {e}")

            if task_index < 0 or task_index >= len(tasks):
                raise ValueError(f"Invalid task index: {task_index} (valid range: 0-{len(tasks)-1})")

            task = tasks[task_index]
            instructions = task.get('instructions', [])

            if not instructions:
                raise ValueError("Task has no instructions")
            
            # Validate instructions format
            if not isinstance(instructions, list):
                raise ValueError(f"Instructions must be a list, got {type(instructions)}")
            
            print(f"[DEBUG] Task: {task.get('name', 'Unknown')}, Instructions type: {type(instructions)}, Length: {len(instructions) if isinstance(instructions, list) else 'N/A'}")

            # Get grid_categories from task
            grid_categories = task.get('grid_categories', ['basic'])
            if not isinstance(grid_categories, list):
                grid_categories = ['basic']

            # Extract parameters from task - new format: list of tags
            # Count parameters in program to determine how many tags we need
            program_str = task.get('program', '')
            num_parameters = count_parameters_in_program(program_str)
            
            task_params = task.get('parameters') or task.get('parameter')
            parameter_tags = convert_parameters_to_tag_list(task_params, num_parameters)

            # Get min_grid_dim and max_grid_dim from task
            min_grid_dim = task.get('min_grid_dim')
            max_grid_dim = task.get('max_grid_dim')

            # Generate examples using the reusable function
            examples = generate_grid_examples(instructions, num_examples, grid_categories, parameters=parameter_tags, min_grid_dim=min_grid_dim, max_grid_dim=max_grid_dim)

            if not examples:
                raise ValueError("Failed to generate any examples")

            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response_data = json.dumps({
                'status': 'success',
                'examples': examples
            })
            self.wfile.write(response_data.encode('utf-8'))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in handle_generate_examples: {error_trace}")
            
            # Make sure we send JSON, not HTML
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = json.dumps({
                'status': 'error',
                'message': str(e),
                'traceback': error_trace
            })
            self.wfile.write(error_response.encode('utf-8'))

    def handle_execute_program(self):
        """Handle request to execute a program string and generate examples."""
        try:
            # Read POST data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                raise ValueError("No content provided")
            
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}")

            # Extract parameters
            program_str = data.get('program', '')
            if not program_str:
                raise ValueError("Program string is required")
            
            grid_categories = data.get('grid_categories', ['basic'])
            if not isinstance(grid_categories, list):
                grid_categories = ['basic']
            
            num_examples = int(data.get('num_examples', 3))

            # Extract parameters from POST data - new format: list of tags
            # Count parameters in program to determine how many tags we need
            num_parameters = count_parameters_in_program(program_str)
            
            post_params = data.get('parameters')
            parameter_tags = convert_parameters_to_tag_list(post_params, num_parameters)

            # Get min_grid_dim and max_grid_dim from POST data (optional)
            min_grid_dim = data.get('min_grid_dim')
            max_grid_dim = data.get('max_grid_dim')

            # Parse program string to hand-written format
            N = len(DSL.semantics)
            try:
                program_lines = block_of_text_to_program_lines(program_str)
                # Convert to instructions
                instructions = ProgUtils.convert_user_format_to_token_seq(program_lines)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                raise ValueError(f"Failed to parse program: {e}\n{error_trace}")

            if not instructions:
                raise ValueError("Program parsed to empty instructions")

            # Generate examples using the reusable function
            examples = generate_grid_examples(instructions, num_examples, grid_categories, parameters=parameter_tags, min_grid_dim=min_grid_dim, max_grid_dim=max_grid_dim)

            if not examples:
                raise ValueError("Failed to generate any examples")

            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response_data = json.dumps({
                'status': 'success',
                'examples': examples
            })
            self.wfile.write(response_data.encode('utf-8'))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in handle_execute_program: {error_trace}")
            
            # Make sure we send JSON, not HTML
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = json.dumps({
                'status': 'error',
                'message': str(e),
                'traceback': error_trace
            })
            self.wfile.write(error_response.encode('utf-8'))

    def serve_file(self, filepath, content_type):
        """Serve a file with the given content type."""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")

    def get_content_type(self, filepath):
        """Determine content type based on file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        content_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }
        return content_types.get(ext, 'application/octet-stream')

    def log_message(self, format, *args):
        """Override to customize log format."""
        print(f"[{self.address_string()}] {format % args}")

def run_server(port=8000):
    """Run the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, TaskManagerHandler)
    print(f"Server running at http://localhost:{port}/")
    print(f"Open http://localhost:{port}/manager_ui/task_manager.html in your browser")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()

