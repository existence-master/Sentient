import os

# List of file names to exclude (add full file names here, e.g., 'file1.py', 'file2.js')
exclude_files = ['.env', '.env.template', ".prettierrc", "eslint.config.js", "jsconfig.json", "next.config.js", "package-lock.json", "package.json", "postcss.config.js", "README.md", "tailwind.config.js", "chatsDb.json", "userProfileDb.json", "requirements.txt", "token.pickle", "run_servers.sh", "version.txt", "collect_code.py"]  # Specify the files to exclude

# List of folder names to exclude (add folder names here, e.g., 'folder1', 'folder2')
exclude_dirs = ['node_modules', '.next', 'public', 'styles', 'input', 'venv', '__pycache__', 'chroma_db',
                'agents', 'scraper']  # Specify the folders to exclude

def get_code_from_files(directory, exclude_files, exclude_dirs):
    all_code = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Remove any excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Exclude files that are in the exclude list
            if file in exclude_files:
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
    code = get_code_from_files(current_directory, exclude_files, exclude_dirs)
    
    # Write the code to an output text file
    output_file = "collected_code.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(code)
    
    print(f"Code from files has been written to {output_file}")

if __name__ == "__main__":
    main()
