#!/bin/bash

# Update your system.
sudo apt update -y

# Install prerequisite packages.
sudo apt install apt-transport-https ca-certificates gnupg -y

# Add the Google Cloud repository.
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Download and add the GPG key.
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Update the package list again.
sudo apt update -y

# Install the Google Cloud CLI.
sudo apt install google-cloud-cli -y

# Verify the installation.
gcloud --version
