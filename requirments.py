# write python code to run installation of packages
import subprocess
import sys

def install_packages(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    
    
packages = ["chromadb", "openai", "pymupdf", "pypdf", "dotenv","flask"]

install_packages(packages)

print("Packages installed successfully")

# write python code to run installation of packages