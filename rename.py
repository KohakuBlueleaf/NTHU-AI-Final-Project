import os


def rename_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if "preferation" in file:
                new_name = file.replace("preferation", "preference")
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed '{old_path}' to '{new_path}'")


# Replace 'your_directory_path' with the path to the directory you want to process.
rename_files(".")
