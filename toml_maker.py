import toml

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Remove comments and empty lines from requirements
requirements = [req for req in requirements if req and not req.startswith("#")]

# Create a dictionary with project metadata
project_metadata = {
    "project": {
        "name": "My EDA and ML Project on Google Colab",  # Replace with your project name
        "version": "0.1.0",  
        "description": "Predict whether a contacted person would agree to deposit money to the bank",  
        "authors": [
            {"name": "Maripjan Koshmatov", "mkoshmatov@gmail.com": "your.email@example.com"},  # Replace with your information
        ],
        "dependencies": requirements,
    },
    "build-system": {
        "requires": ["setuptools", "wheel"],
        "build-backend": "setuptools.build_meta",
    },
}

# Write the project metadata to pyproject.toml
with open("pyproject.toml", "w") as f:
    toml.dump(project_metadata, f)

print("pyproject.toml file created successfully!")