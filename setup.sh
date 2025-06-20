#!/bin/bash
# setup.sh

# Install dependencies with build optimizations
pip install --no-cache-dir -r requirements.txt

# Reduce package size
find /usr/local/lib/python3.*/ -name '*.pyc' -delete
find /usr/local/lib/python3.*/ -type d -name '__pycache__' -exec rm -rf {} +
rm -rf /root/.cache/pip