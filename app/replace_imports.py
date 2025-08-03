import os

root_dir = "app"

for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".py"):
            filepath = os.path.join(subdir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            new_content = content.replace("from app.config import", "from app.config import")
            if content != new_content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"Updated {filepath}")
