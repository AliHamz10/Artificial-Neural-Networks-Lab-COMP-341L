# Lab Report 1: Installation and Setup of Cursor IDE

**Course:** COMP-341L - Artificial Neural Networks Lab  
**Lab Number:** 1  
**Date:** [Date]  
**Student:** [Your Name]

---

## Objective

The objective of this lab is to install and configure Cursor IDE, a modern code editor designed for AI-assisted development, and set it up for the Artificial Neural Networks Lab course.

---

## Introduction

Cursor is an AI-powered code editor built on VS Code that provides intelligent code completion, AI chat, and advanced editing features. This lab report documents the step-by-step installation process and initial configuration of Cursor IDE for use in the COMP-341L course.

---

## Prerequisites

- macOS/Windows/Linux operating system
- Internet connection
- Administrator/sudo privileges (for installation)

---

## Installation Steps

### Step 1: Download Cursor

1. Open your web browser and navigate to the official Cursor website: [https://cursor.sh](https://cursor.sh)
2. Click on the "Download" button
3. Select the appropriate version for your operating system:
   - **macOS**: Download the `.dmg` file
   - **Windows**: Download the `.exe` installer
   - **Linux**: Download the `.AppImage` or `.deb`/.`rpm` package

**Screenshot:** `screenshots/01_download_page.png`

---

### Step 2: Install Cursor (macOS)

1. Locate the downloaded `.dmg` file in your Downloads folder
2. Double-click the `.dmg` file to open it
3. Drag the Cursor icon to the Applications folder
4. Wait for the copy process to complete
5. Eject the disk image

**Screenshot:** `screenshots/02_install_macos.png`

---

### Step 2: Install Cursor (Windows)

1. Locate the downloaded `.exe` installer file
2. Double-click the installer to launch it
3. Follow the installation wizard:
   - Accept the license agreement
   - Choose installation location (default recommended)
   - Select additional options if needed
4. Click "Install" and wait for the installation to complete
5. Click "Finish" when done

**Screenshot:** `screenshots/02_install_windows.png`

---

### Step 2: Install Cursor (Linux)

**For .deb packages (Ubuntu/Debian):**
```bash
sudo dpkg -i cursor_*.deb
sudo apt-get install -f  # Install dependencies if needed
```

**For .rpm packages (Fedora/RHEL):**
```bash
sudo rpm -i cursor_*.rpm
```

**For AppImage:**
```bash
chmod +x cursor_*.AppImage
./cursor_*.AppImage
```

**Screenshot:** `screenshots/02_install_linux.png`

---

### Step 3: Launch Cursor

1. Open Cursor from your Applications folder (macOS) or Start Menu (Windows)
2. On first launch, you may see a welcome screen
3. Cursor will initialize and open the editor interface

**Screenshot:** `screenshots/03_first_launch.png`

---

### Step 4: Sign In / Create Account

1. If prompted, sign in to your Cursor account or create a new one
2. You can use:
   - Email sign-up
   - GitHub account (recommended)
   - Google account
3. Complete the authentication process

**Screenshot:** `screenshots/04_sign_in.png`

---

### Step 5: Initial Setup and Configuration

1. **Choose a theme**: Select your preferred color theme (Light/Dark)
2. **Install recommended extensions** (optional):
   - Python extension (for Python development)
   - Jupyter extension (for notebook support)
   - Git extension (for version control)
3. **Configure settings**:
   - Open Settings (Cmd/Ctrl + ,)
   - Adjust editor preferences as needed
   - Set up font size, tab size, etc.

**Screenshot:** `screenshots/05_settings.png`

---

### Step 6: Install Python and Required Tools

1. **Check Python installation**:
   ```bash
   python3 --version
   ```

2. **Install Python** (if not installed):
   - macOS: `brew install python3` or download from python.org
   - Windows: Download from python.org
   - Linux: `sudo apt-get install python3 python3-pip`

3. **Install required packages**:
   ```bash
   pip3 install numpy pandas matplotlib scikit-learn tensorflow torch jupyter
   ```

**Screenshot:** `screenshots/06_python_setup.png`

---

### Step 7: Configure Git (if not already configured)

1. **Set up Git identity**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **Verify Git installation**:
   ```bash
   git --version
   ```

**Screenshot:** `screenshots/07_git_config.png`

---

### Step 8: Open Project Folder

1. In Cursor, click "File" → "Open Folder"
2. Navigate to your project directory:
   ```
   /Users/alihamza/CursorCode Projects/Artificial-Neural-Networks-Lab-COMP-341L
   ```
3. Click "Open"

**Screenshot:** `screenshots/08_open_project.png`

---

### Step 9: Verify Installation

1. **Check Cursor version**:
   - Go to "Cursor" → "About Cursor" (macOS) or "Help" → "About" (Windows/Linux)
   - Note the version number

2. **Test AI features**:
   - Open a new file
   - Try the AI chat feature (Cmd/Ctrl + L)
   - Test code completion

**Screenshot:** `screenshots/09_verify_installation.png`

---

## Configuration Summary

### Installed Components

- ✅ Cursor IDE: Version [Version Number]
- ✅ Python: Version [Version Number]
- ✅ Git: Version [Version Number]
- ✅ Required Python packages: numpy, pandas, matplotlib, scikit-learn, tensorflow, torch, jupyter

### Settings Configured

- Editor theme: [Your Theme]
- Font size: [Your Font Size]
- Tab size: [Your Tab Size]
- Line endings: [LF/CRLF]
- Python interpreter: [Path to Python]

---

## Troubleshooting

### Common Issues and Solutions

1. **Cursor won't launch**
   - **Solution**: Check system requirements and ensure you have the latest OS updates

2. **Python not recognized**
   - **Solution**: Add Python to PATH or use full path to python3 executable

3. **Git authentication issues**
   - **Solution**: Set up SSH keys or use HTTPS with personal access token

4. **Extensions not loading**
   - **Solution**: Restart Cursor and check extension compatibility

---

## Conclusion

Cursor IDE has been successfully installed and configured for the Artificial Neural Networks Lab course. The editor is ready for development work with AI-assisted features enabled. All required tools (Python, Git) have been verified and configured.

---

## References

- Cursor Official Website: https://cursor.sh
- Cursor Documentation: https://docs.cursor.sh
- Python Official Website: https://www.python.org
- Git Documentation: https://git-scm.com/doc

---

## Appendix

### Screenshots Index

1. `screenshots/01_download_page.png` - Cursor download page
2. `screenshots/02_install_[os].png` - Installation process
3. `screenshots/03_first_launch.png` - First launch screen
4. `screenshots/04_sign_in.png` - Sign in screen
5. `screenshots/05_settings.png` - Settings configuration
6. `screenshots/06_python_setup.png` - Python installation verification
7. `screenshots/07_git_config.png` - Git configuration
8. `screenshots/08_open_project.png` - Opening project folder
9. `screenshots/09_verify_installation.png` - Verification of installation

---

**Report prepared by:** [Your Name]  
**Date:** [Date]
