#!/bin/bash
# Install Grafial extension to Cursor manually

EXT_DIR="$HOME/.cursor/extensions/grafial-0.1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Grafial extension to Cursor..."
echo "Target directory: $EXT_DIR"

# Create extensions directory if it doesn't exist
mkdir -p "$HOME/.cursor/extensions"

# Copy extension files
cp -r "$SCRIPT_DIR" "$EXT_DIR"

# Remove the install script and .vsix from the copied extension
rm -f "$EXT_DIR/install-cursor.sh"
rm -f "$EXT_DIR/*.vsix"

echo "✓ Extension installed to $EXT_DIR"
echo ""
echo "Please restart Cursor for the extension to take effect."
echo "Open a .grafial file to verify syntax highlighting is working."

