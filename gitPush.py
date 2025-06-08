# import subprocess
# import time

# def git_push():
#     while True:
#         try:
#             # Run the git push command
#             result = subprocess.run(['git', 'push'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             # If the push was successful, break the loop
#             print(result.stdout.decode())
#             print("Push successful!")
#             break
#         except subprocess.CalledProcessError as e:
#             # Print the error message and wait before retrying
#             print(e.stderr.decode())
#             print("Failed to push. Retrying in 30 seconds...")
#             time.sleep(30)

# if __name__ == "__main__":
#     git_push()
import subprocess
import time

def git_push():
    # Get commit message from the user
    commit_message = input("Enter the commit message: ")

        # Run the pip freeze command and write to requirements.txt
    with open('requirements.txt', 'w') as f:
        subprocess.run(['pip', 'freeze'], stdout=f, check=True)
    print("Requirements.txt updated.")
    
    # Run the git add command
    subprocess.run(['git', 'add', '.'], check=True)
    print("Files added to staging area.")

    # Run the git commit command with the provided commit message
    subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    print("Commit successful.")

    while True:
        try:
            # Run the git push command
            result = subprocess.run(['git', 'push'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
            print("Push successful!")
            break
        except subprocess.CalledProcessError as e:
            # Print the error message and wait before retrying
            print(e.stderr.decode())
            print("Failed to push. Retrying in 15 seconds...")
            time.sleep(15)

if __name__ == "__main__":
    git_push()

# useage: python gitPush.py