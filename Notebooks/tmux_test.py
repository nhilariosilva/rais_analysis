import time

# Open a file named 'output.txt' in write mode ('w')
with open('output.txt', 'w') as file:
    # Loop 60 times (from 1 to 60)
    for i in range(1, 61):
        time.sleep(1)
        # Write the current iteration number followed by a newline character
        file.write(f"{i}\n")

# The 'with' statement automatically handles closing and saving the file
print("File 'output.txt' has been created with 60 lines.")
