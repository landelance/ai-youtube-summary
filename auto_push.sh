#!/bin/bash

# Navigate to your project folder
cd /path/to/your/folder

# Add all new/modified files
git add .

# Commit with timestamp
git commit -m "Auto-commit on $(date '+%Y-%m-%d %H:%M:%S')"

# Push to the main branch
git push origin main
