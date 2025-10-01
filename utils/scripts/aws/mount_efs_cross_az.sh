#!/usr/bin/env bash
set -euo pipefail

efs_ids=( "fs-06c351289107625f5" "fs-09e44a2da46a4d4f9" )
efs_mount_paths=( "/mnt/efs/all-cxr"    "/mnt/efs/models"          )
aws_region="us-east-2"

if [ "${#efs_ids[@]}" -ne "${#efs_mount_paths[@]}" ]; then
  echo "Error: efs_ids and efs_mount_paths lengths differ" >&2
  exit 1
fi

echo "→ Enabling universe (so we can pull amazon-efs-utils if available)…"
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt-get update -y

echo "→ Installing prerequisites: nfs-common, stunnel4…"
sudo apt-get install -y nfs-common stunnel4

if apt-cache show amazon-efs-utils &>/dev/null; then
  echo "→ Installing amazon-efs-utils from Ubuntu repo…"
  sudo apt-get install -y amazon-efs-utils
else
  echo "→ amazon-efs-utils not in repo → falling back to build-from-source"
  # ensure cargo & build tools
  sudo apt-get install -y curl build-essential pkg-config libssl-dev
  # install Rustup/Cargo
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  export PATH="$HOME/.cargo/bin:$PATH"
  # build
  TMP=/tmp/efs-utils
  git clone https://github.com/aws/efs-utils.git "$TMP"
  cd "$TMP"
  ./build-deb.sh
  deb=$(ls build/amazon-efs-utils-*.deb | head -n1)
  sudo dpkg -i "$deb"
  sudo apt-get install -f -y     # fix any missing deps
  cd -
  rm -rf "$TMP"
fi

echo ""
echo "→ Mounting file systems…"
for i in "${!efs_ids[@]}"; do
  id="${efs_ids[$i]}"
  # if this is the one zone file system, add the az override
  extra_opts=""
  if [[ "$id" == "fs-06c351289107625f5" ]]; then
    extra_opts=",az=us-east-2a"
  fi
  target="${efs_mount_paths[$i]}"

  echo "  • Creating mount point $target"
  sudo mkdir -p "$target"

  echo "  • Mounting $id → $target"
  sudo mount -t efs -o tls,region="$aws_region"$extra_opts "$id":/ "$target"

  echo "  • Adding to /etc/fstab if needed"
  fstab_line="$id:/ $target efs _netdev,tls,region=$aws_region$extra_opts 0 0"
  if ! grep -Fqs "$fstab_line" /etc/fstab; then
    echo "$fstab_line" | sudo tee -a /etc/fstab
  fi

  echo "  ✔ $id mounted successfully"
done

echo "All done!"
