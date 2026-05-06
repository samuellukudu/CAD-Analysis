# Fonts Directory

This directory is used to store project-local fonts for consistent rendering across different environments (macOS, Linux, Windows).

## Recommended Fonts
For best results with Chinese (Simplified) DXF files, download and place the following fonts here:

- **NotoSansCJKsc-Regular.otf**
- **NotoSansCJKsc-Medium.otf**

## Linux Server Deployment
On a Linux server, you can either:
1. Install the system packages (e.g., `sudo apt-get install fonts-noto-cjk`).
2. Or manually place the `.otf` files in this directory. The code is configured to check this folder first.
