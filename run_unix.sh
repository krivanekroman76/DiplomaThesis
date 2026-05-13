#!/bin/bash
# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    ./.venv/bin/pip install --default-timeout=1000 -r requirements.txt
fi

echo "Starting Separation App..."
./.venv/bin/python3 separation_app.py
read -p "Press enter to exit..."