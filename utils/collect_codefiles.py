import os

# List of file names to exclude
exclude_files = ['.env', '.env.template', ".prettierrc", "eslint.config.js", "jsconfig.json", "next.config.js", "package-lock.json", "package.json", "postcss.config.js", "README.md", "tailwind.config.js", "chatsDb.json", "userProfileDb.json", "requirements.txt", "token.pickle", "run_servers.sh", "version.txt", "collect_code.py", "prompts.py", "formats.py", "runnables.py", "functions.py", "helpers.py", "externals.py", "constants.py"]

# List of folders to exclude
exclude_dirs = ['node_modules', '.next', 'public', 'styles', 'input', 'venv', '__pycache__', 'chroma_db']

# List of specific folders to include
include_folders = ['chat', 'auth', 'agents', 'app', 'common', 'memory', 'scraper', 'utils']

# List of specific files to include
include_files = []  # Leave empty to include all files in the specified folders

def get_code_from_files(directory, exclude_files, exclude_dirs, include_folders, include_files):
    all_code = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Remove any excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Get the relative path for the current directory
        relative_root = os.path.relpath(root, directory)
        print(f"Checking directory: {relative_root}")  # Debug statement to check the directory paths
        
        # Check if the folder is one of the included folders or subfolders
        is_included = any(relative_root.startswith(include_folder) for include_folder in include_folders)
        
        # Skip this directory if it's not in any of the included folders
        if not is_included:
            continue
        
        # Process each file in the current directory
        for file in files:
            # Exclude files that are in the exclude list
            if file in exclude_files:
                continue
            
            # If include_files is not empty, only consider the specified files
            if include_files and os.path.relpath(file, root) not in include_files:
                continue

            # Full path of the file
            file_path = os.path.join(root, file)
            
            # Read the file and append its content with the file path
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    # Add the relative file path and code to the result
                    relative_path = os.path.relpath(file_path, directory)
                    all_code.append(f"### {relative_path} ###\n{code}\n")
            except Exception as e:
                print(f"Couldn't read {file_path}: {e}")
    
    return all_code


def main():
    # Get the current working directory
    current_directory = "/home/existence/Desktop/Sentient/Open Source/SK/development/src/model"
    
    # Get all the code from the files in the specified directory
    code = get_code_from_files(current_directory, exclude_files, exclude_dirs, include_folders, include_files)
    
    # Write the code to an output text file
    output_file = "collected_code_from_selected_folders.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(code)
    
    print(f"Code from files has been written to {output_file}")

if __name__ == "__main__":
    main()
