import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)

# Get the root directory of the project
root_directory = os.path.dirname(current_directory)

# Print the root directory
print("Root Directory:", root_directory)
