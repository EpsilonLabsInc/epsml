#!/bin/bash

# User defined variables.
username="andrej"
ssh_key="./keys/andrej_eps_ssh_key"
ssh_key_pub="./keys/andrej_eps_ssh_key.pub"
gcp_service_key="./keys/andrej-service-account-key.json"
git_name_of_user="Andrej Ikica"
git_user_email="andrej@epsilonlabs.ai"
apply_intern_vl_code_changes=1
conda_env_name="ai"
python_ver="3.11"
mlflow_username=""
mlflow_password=""
wandb_api_key=""
hugging_face_token=""
aws_access_key_id=""
aws_secret_access_key=""
aws_session_token=""

# File names and directories used throughout the installation.
script_dir=$(dirname "$(readlink -f "$0")")
temp_dir=$(mktemp -d)
home_dir="/home/$username"
profile_file="$home_dir/.bashrc"
ssh_dir="$home_dir/.ssh"
ssh_config_file="$ssh_dir/config"
keys_dir="$home_dir/keys"
miniconda_installer="$temp_dir/miniconda_installer.sh"
miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
miniconda_install_path="$home_dir/apps/miniconda3"
git_root_dir="$home_dir/work"
epsutils_dir="$git_root_dir/epsutils"
epsdatasets_dir="$git_root_dir/epsdatasets"
two_dee_image_encoders_dir="$git_root_dir/2d-image-encoders"
epsclassifiers_dir="$git_root_dir/epsclassifiers"
reports_pipeline_dir="$git_root_dir/reports-pipeline"
dinov2_dir="$git_root_dir/dinov2-torch_2_1"
dinov3_dir="$git_root_dir/dinov3"
internvl_dir="$git_root_dir/InternVL"

# --------------------------------------------------------
# Home directory.
# --------------------------------------------------------

if [ ! -d "/home/$username" ]; then
  echo "The home directory for user '$username' does not exist"
  exit 1
fi

# --------------------------------------------------------
# SSH keys.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "SSH keys"
echo "----------------------------------------"
echo ""

if [ ! -d "$ssh_dir" ]; then
  mkdir -p "$ssh_dir"
  echo "Created directory $ssh_dir"
fi

chmod 700 "$ssh_dir"

for file in "$ssh_key" "$ssh_key_pub"; do
  if [ -f "$script_dir/$file" ]; then
    cp "$script_dir/$file" "$ssh_dir/"
    echo "Copied $file to $ssh_dir"
    base_name=$(basename "$file")
    chmod 600 "$ssh_dir/$base_name"
  else
    echo "File $file not found in $script_dir"
    exit 1
  fi
done

if [ $? -eq 0 ]; then
  echo "SSH keys successfully copied to $ssh_dir"
else
  echo "An error occurred copying SSH keys to $ssh_dir"
  exit 1
fi

# --------------------------------------------------------
# SSH config setup.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "SSH config setup"
echo "----------------------------------------"
echo ""

echo "Host *
  AddKeysToAgent yes
  IdentityFile $ssh_dir/andrej_eps_ssh_key" > "$ssh_config_file"

chmod 600 "$ssh_config_file"

if [ $? -eq 0 ]; then
  echo "SSH config file $ssh_config_file successfully created"
else
  echo "An error occurred creating SSH config file $ssh_config_file"
  exit 1
fi

# --------------------------------------------------------
# GCP service key.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "GCP service key"
echo "----------------------------------------"
echo ""

if [ ! -d "$keys_dir" ]; then
  mkdir -p "$keys_dir"
  echo "Created directory $keys_dir"
fi

chmod 700 "$keys_dir"

if [ -f "$script_dir/$gcp_service_key" ]; then
  cp "$script_dir/$gcp_service_key" "$keys_dir/"
  echo "Copied $gcp_service_key to $keys_dir"
  base_name=$(basename "$gcp_service_key")
  chmod 600 "$keys_dir/$base_name"
else
  echo "File $gcp_service_key not found in $keys_dir"
  exit 1
fi

if [ $? -eq 0 ]; then
  echo "GCP service key successfully copied to $keys_dir"
else
  echo "An error occurred copying GCP service key to $keys_dir"
  exit 1
fi

# --------------------------------------------------------
# Git.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Git"
echo "----------------------------------------"
echo ""

if ! command -v git &> /dev/null; then
  echo "Git is not installed, please install Git and try again"
  exit 1
fi

# Add GitHub's public SSH key to known_hosts.
ssh-keyscan github.com >> "$ssh_dir/known_hosts"

git config --global user.name "$git_name_of_user"
git config --global user.email "$git_user_email"

if [ ! -d "$git_root_dir" ]; then
  mkdir -p "$git_root_dir"
  echo "Created Git root directory $git_root_dir"
fi

chmod 700 "$git_root_dir"

if [ ! -d "$epsutils_dir" ]; then
  echo "Cloning epsutils"
  git clone "git@github.com:EpsilonLabsInc/epsutils.git" "$epsutils_dir"
else
  echo "Git repo epsutils already exists"
fi

if [ ! -d "$epsdatasets_dir" ]; then
  echo "Cloning epsdatasets"
  git clone "git@github.com:EpsilonLabsInc/epsdatasets.git" "$epsdatasets_dir"
else
  echo "Git repo epsdatasets already exists"
fi

if [ ! -d "$two_dee_image_encoders_dir" ]; then
  echo "Cloning 2d-image-encoders"
  git clone "git@github.com:EpsilonLabsInc/2d-image-encoders.git" "$two_dee_image_encoders_dir"
else
  echo "Git repo 2d-image-encoders already exists"
fi

if [ ! -d "$epsclassifiers_dir" ]; then
  echo "Cloning epsclassifiers"
  git clone "git@github.com:EpsilonLabsInc/epsclassifiers.git" "$epsclassifiers_dir"
else
  echo "Git repo epsclassifiers already exists"
fi

if [ ! -d "$reports_pipeline_dir" ]; then
  echo "Cloning reports-pipeline"
  git clone "git@github.com:EpsilonLabsInc/reports-pipeline.git" "$reports_pipeline_dir"
else
  echo "Git repo reports-pipeline already exists"
fi

if [ ! -d "$dinov2_dir" ]; then
  echo "Cloning dinov2-torch_2_1"
  git clone "git@github.com:EpsilonLabsInc/dinov2-torch_2_1.git" "$dinov2_dir"
else
  echo "Git repo dinov2-torch_2_1 already exists"
fi

if [ ! -d "$dinov3_dir" ]; then
  echo "Cloning dinov3"
  git clone "git@github.com:EpsilonLabsInc/dinov3.git" "$dinov3_dir"
else
  echo "Git repo dinov3 already exists"
fi

if [ ! -d "$internvl_dir" ]; then
  echo "Cloning InternVL"
  git clone "https://github.com/OpenGVLab/InternVL.git" "$internvl_dir"
else
  echo "Git repo InternVL already exists"
fi

if [ $? -eq 0 ]; then
  echo "Git repos successfully cloned to $git_root_dir"
else
  echo "Error cloning Git repos to $git_root_dir"
  exit 1
fi

if [[ $apply_intern_vl_code_changes -ne 0 ]]; then
  echo "Applying InternVL code changes"

  target_line_number=347
  old_content="                    hidden_states)"
  new_content="                    hidden_states, use_reentrant=False)"
  file_path="$internvl_dir/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py"
  current_content=$(sed "${target_line_number}q;d" "$file_path")

  if [[ "$current_content" == "$old_content" ]]; then
      sed -i "${target_line_number}s/.*/$new_content/" "$file_path"
      echo "Line $target_line_number was replaced successfully"
  else
      echo "The content of the line $target_line_number doesn't match the expected content, no changes made"
  fi
fi

# --------------------------------------------------------
# Miniconda.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Miniconda"
echo "----------------------------------------"
echo ""

echo "Downloading Miniconda installer to $miniconda_installer"
curl -o "$miniconda_installer" "$miniconda_url"

echo "Installing Miniconda to $miniconda_install_path"
bash "$miniconda_installer" -b -p "$miniconda_install_path"

echo "Running 'conda init'"
"$miniconda_install_path/bin/conda" init

echo "Deleting installer script and temp dir"
rm -rf "$miniconda_installer"
rm -rf "$temp_dir"

echo "Installation completed. Miniconda installed at $miniconda_install_path"

# --------------------------------------------------------
# Conda env.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Conda env"
echo "----------------------------------------"
echo ""

# Source Conda setup directly from install path.
source "$miniconda_install_path/etc/profile.d/conda.sh"

# Accept the terms of service before creating the environment.
"$miniconda_install_path/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$miniconda_install_path/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name "$conda_env_name" python="$python_ver" -y

if [ $? -eq 0 ]; then
  echo "Conda environment successfully created"
else
  echo "Error creating Conda environment"
  exit 1
fi

echo "Activating Conda environment"
conda activate "$conda_env_name"

# --------------------------------------------------------
# Python packages.
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Python packages"
echo "----------------------------------------"
echo ""

echo "Installing all the required Python packages"

pip install -r "$epsutils_dir/requirements.txt"
pip install -r "$epsdatasets_dir/requirements.txt"
pip install -r "$two_dee_image_encoders_dir/requirements.txt"
pip install -r "$epsclassifiers_dir/requirements.txt"
pip install -r "$reports_pipeline_dir/requirements.txt"
pip install -r "$dinov3_dir/requirements.txt"

echo "Python packages installed"

# --------------------------------------------------------
# Environment variables
# --------------------------------------------------------

echo ""
echo "----------------------------------------"
echo "Environment variables"
echo "----------------------------------------"
echo ""

echo "Writing environment variables"

echo "" >> "$profile_file"
echo "# Epsilon development environment variables" >> "$profile_file"
echo "" >> "$profile_file"

echo "export PYTHONPATH=\"$epsutils_dir:$epsdatasets_dir:$two_dee_image_encoders_dir:$epsclassifiers_dir:$reports_pipeline_dir:$dinov2_dir:$dinov3_dir:$internvl_dir/internvl_chat:\$PYTHONPATH\"" >> "$profile_file"
echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$keys_dir/$gcp_service_key\"" >> "$profile_file"
echo "export MLFLOW_TRACKING_USERNAME=\"$mlflow_username\"" >> "$profile_file"
echo "export MLFLOW_TRACKING_PASSWORD=\"$mlflow_password\"" >> "$profile_file"
echo "export WANDB_API_KEY=\"$wandb_api_key\"" >> "$profile_file"
echo "export HF_TOKEN=\"$hugging_face_token\"" >> "$profile_file"
echo "export AWS_ACCESS_KEY_ID=\"$aws_access_key_id\"" >> "$profile_file"
echo "export AWS_SECRET_ACCESS_KEY=\"$aws_secret_access_key\"" >> "$profile_file"
echo "export AWS_SESSION_TOKEN=\"$aws_session_token\"" >> "$profile_file"

echo "Environment variables written"

# --------------------------------------------------------
# Finalization.
# --------------------------------------------------------

echo ""
echo "Environment setup complete. Please start a new shell session and run 'conda activate $conda_env_name' to activate the Conda environment."
echo ""
