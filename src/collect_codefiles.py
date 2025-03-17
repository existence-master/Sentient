import os

# List of file names to exclude (can still be specified as in the original script)
exclude_files = ['.env', '.env.template', ".prettierrc", "eslint.config.js", "jsconfig.json", "next.config.js", "package-lock.json", "package.json", "postcss.config.js", "README.md", "tailwind.config.js", "chatsDb.json", "userProfileDb.json", "requirements.txt", "token.pickle", "run_servers.sh", "version.txt", "collect_code.py"]

# List of folders to exclude (can still be specified as in the original script)
exclude_dirs = ['node_modules', '.next', 'public', 'styles', 'input', 'venv', '__pycache__', 'chroma_db', 'agents', 'scraper']

# List of specific folders to include (this will contain the folder names or relative paths to include)
include_folders = ['/src/interface/chat', '/src/model/chat']  # Example specific paths or folder names

# List of specific files to include (optional - can specify exact file paths if needed)
include_files = []  # Leave empty to include all files in the specified folders

def get_code_from_files(directory, exclude_files, exclude_dirs, include_folders, include_files):
    all_code = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Remove any excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Check if the folder is one of the included folders
        relative_root = os.path.relpath(root, directory)
        
        # Skip if the current directory doesn't match any of the include folders
        if not any(include_folder in relative_root for include_folder in include_folders):
            continue
        
        for file in files:
            # Exclude files that are in the exclude list
            if file in exclude_files:
                continue
            
            # If include_files is not empty, only consider the specified files
            if include_files and file not in include_files:
                continue

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
    current_directory = os.getcwd()
    
    # Get all the code from the files in the specified directory
    code = get_code_from_files(current_directory, exclude_files, exclude_dirs, include_folders, include_files)
    
    # Write the code to an output text file
    output_file = "collected_code_from_selected_folders.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(code)
    
    print(f"Code from files has been written to {output_file}")

if __name__ == "__main__":
    main()
