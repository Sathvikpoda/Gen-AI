import os

# Specify the file name with the full path
file_name = r"C:\Users\sathv\Gen-AI\sample\examplefile.txt"  # Change this to your file path

# Function to split text into smaller paragraphs
def split_into_paragraphs(text, max_length=1000):
    paragraphs = []
    while len(text) > max_length:
        # Find the last space before the max_length to avoid splitting words
        split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:  # No space found, split at max_length
            split_point = max_length
        paragraphs.append(text[:split_point].strip())
        text = text[split_point:].strip()
    
    # Append any remaining text as the last paragraph
    if text:
        paragraphs.append(text)
    
    return paragraphs

try:
    # Attempt to open the file in read mode
    with open(file_name, 'r') as file:
        # Read the content of the file
        content = file.read()
        # Print the content of the file
        print("File content:\n")
        print(content)

except FileNotFoundError:
    # If file not found, ask the user for custom content
    print(f"The file '{file_name}' was not found.")
    
    # Prompt user to enter content to write to the new file
    user_content = input("Please enter the content you want to write to the new file: ")
    
    # Split the user content into smaller paragraphs
    paragraphs = split_into_paragraphs(user_content, max_length=1000)
    
    # Create and write the chunked paragraphs to the file
    with open(file_name, 'w') as file:
        for paragraph in paragraphs:
            file.write(paragraph + '\n\n')  # Separate paragraphs by double new line
    
    print(f"File '{file_name}' created with the following content (split into small paragraphs):\n")
    for paragraph in paragraphs:
        print(paragraph)
        print("\n")

except PermissionError:
    print(f"Permission denied: Unable to read or write the file '{file_name}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
