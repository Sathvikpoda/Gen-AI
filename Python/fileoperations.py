# Specify the file name with the full path
file_name = r"C:\Users\sathv\Gen-AI\sample\test.py"  # Change this to your file path

try:
    # Open the file in read mode
    with open(file_name, 'r') as file:
        # Read the content of the file
        content = file.read()
        # Print the content of the file
        print("File content:\n")
        print(content)
except FileNotFoundError:
    print(f"The file '{file_name}' was not found.")
except PermissionError:
    print(f"Permission denied: Unable to read the file '{file_name}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
