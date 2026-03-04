#!/bin/bash
# Run Lava.jl tests on lavapipe (software Vulkan renderer).
# Usage: ./test/run_lavapipe.sh
#
# Requires mesa-vulkan-drivers installed:
#   openSUSE: sudo zypper install Mesa-vulkan-device-select
#   Ubuntu:   sudo apt install mesa-vulkan-drivers
#   Fedora:   sudo dnf install mesa-vulkan-drivers

set -euo pipefail

# Find lavapipe ICD
ICD=""
for path in \
    /usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
    /usr/share/vulkan/icd.d/lvp_icd.json \
    /etc/vulkan/icd.d/lvp_icd.x86_64.json \
    /etc/vulkan/icd.d/lvp_icd.json; do
    if [ -f "$path" ]; then
        ICD="$path"
        break
    fi
done

if [ -z "$ICD" ]; then
    echo "ERROR: lavapipe ICD not found. Install mesa-vulkan-drivers."
    exit 1
fi

echo "Using lavapipe ICD: $ICD"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VK_ICD_FILENAMES="$ICD" \
    julia --project="$PROJECT_DIR" "$SCRIPT_DIR/runtests.jl" "$@"
