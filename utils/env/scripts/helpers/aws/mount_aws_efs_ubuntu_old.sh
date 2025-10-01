#!/bin/bash

# User defined variables.
efs_ids=("fs-06c351289107625f5" "fs-09e44a2da46a4d4f9")  # List of EFS IDs.
efs_mount_paths=("/mnt/efs/all-cxr" "/mnt/efs/models")  # Paths where EFS IDs will be mounted. Sizes of efs_ids and efs_mount_paths must match!
aws_region="us-east-2a"

# --------------------------------------------------------
# Check efs_ids and efs_mount_paths sizes.
# --------------------------------------------------------

len1=${#efs_ids[@]}
len2=${#efs_mount_paths[@]}

if [ "$len1" -ne "$len2" ]; then
  echo "Sizes of efs_ids and efs_mount_paths must match"
  exit 1
fi

# --------------------------------------------------------
# Install build essential.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Build essential"
echo "----------------------------------------"
echo ""

echo "Installing build essential"
sudo apt update && sudo apt install -y build-essential

# --------------------------------------------------------
# Install OpenSSL.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "OpenSSL"
echo "----------------------------------------"
echo ""

echo "Installing OpenSSL"

sudo apt install pkg-config
sudo apt install libssl-dev

export OPENSSL_DIR=/usr/include/openssl
export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu
export OPENSSL_INCLUDE_DIR=/usr/include/openssl
export PKG_CONFIG_PATH=/usr/local/ssl/lib/pkgconfig

# --------------------------------------------------------
# Install Cargo.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Cargo"
echo "----------------------------------------"
echo ""

echo "Installing Cargo package manager"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc
cargo --version

# --------------------------------------------------------
# Install AWS CLI.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "AWS CLI"
echo "----------------------------------------"
echo ""

echo "Installing AWS CLI"
sudo apt install awscli

# --------------------------------------------------------
# Install EFS utils.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "EFS utils"
echo "----------------------------------------"
echo ""

echo "Cloning EFS utils repo"
git clone https://github.com/aws/efs-utils

echo "Building EFS utils"
cd efs-utils
./build-deb.sh

echo "Installing EFS utils"
sudo chmod 644 ./build/amazon-efs-utils-2.2.1-1_amd64.deb
sudo apt-get install -y ./build/amazon-efs-utils*.deb

echo "Removing EFS utils dir"
cd ..
rm -rf efs-utils

# --------------------------------------------------------
# Configure AWS.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "AWS configuration"
echo "----------------------------------------"
echo ""

echo "Configuring AWS"
echo "Use the followng AWS region: $aws_region"
aws configure

# --------------------------------------------------------
# Mount EFS.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "EFS mount"
echo "----------------------------------------"
echo ""

for i in "${!efs_ids[@]}"; do
  efs_id="${efs_ids[$i]}"
  efs_mount_path="${efs_mount_paths[$i]}"

  echo "Creating mount dir $efs_mount_path"
  mkdir -p $efs_mount_path

  echo "Mounting EFS ID $efs_id at $efs_mount_path"
  sudo -E mount -t efs -o tls $efs_id:/ $efs_mount_path

  if [ $? -eq 0 ]; then
    echo "$efs_id successfully mounted"
  else
    echo "An error occurred while mounting $efs_id"
    exit 1
  fi

  echo "Setting up permanent mount"
  line_to_add="$efs_id:/ $efs_mount_path efs tls,_netdev"
  if ! grep -Fxq "$line_to_add" /etc/fstab; then
    echo "Adding the line '$line_to_add' to /etc/fstab"
    echo "$line_to_add" | sudo tee -a /etc/fstab > /dev/null
  else
    echo "The line '$line_to_add' already exists in /etc/fstab"
  fi
done
