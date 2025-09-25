#!/bin/bash

while python3 main.py; [ $? -eq 1 ]; do
    echo "Script exited with code 1. Restarting..."
    sleep 1
done

echo "Script exited with code $?. Exiting."
