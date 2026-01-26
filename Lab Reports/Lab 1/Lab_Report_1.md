# Lab Report 1: Installation and Setup of Cursor IDE

---

**Course Code:** COMP-341L  
**Course Name:** Artificial Neural Networks Lab  
**Lab Number:** 1  
**Lab Title:** Installation and Setup of Cursor IDE  
**Date:** [Date]  

**Name:** [Your Full Name]  
**Roll Number:** [Your Roll Number]  
**Section:** [Your Section]  

---

## Table of Contents

1. [Objective](#objective)
2. [Introduction](#introduction)
3. [Prerequisites](#prerequisites)
4. [Task 1: Downloading Cursor IDE](#task-1-downloading-cursor-ide)
5. [Task 2: Installing Cursor IDE](#task-2-installing-cursor-ide)
6. [Task 3: First Launch and Initial Setup](#task-3-first-launch-and-initial-setup)
7. [Task 4: Configuring Cursor Settings](#task-4-configuring-cursor-settings)
8. [Task 5: Verifying Installation](#task-5-verifying-installation)
9. [Results and Observations](#results-and-observations)
10. [Conclusion](#conclusion)
11. [References](#references)
12. [Appendix: Screenshots](#appendix-screenshots)

---

## Objective

The primary objective of this lab is to successfully download, install, and configure Cursor IDE on the local system. This includes understanding the installation process, configuring initial settings, and verifying that the IDE is properly set up for use in the Artificial Neural Networks Lab course.

---

## Introduction

Cursor is an advanced code editor built on the foundation of Visual Studio Code, enhanced with AI-powered features for intelligent code completion, AI chat assistance, and modern development tools. It provides a seamless coding experience with integrated AI capabilities that help developers write code more efficiently.

This lab report documents the complete installation process of Cursor IDE, including downloading the application, installing it on the system, performing initial configuration, and verifying the successful installation. The report includes detailed step-by-step procedures with corresponding screenshots to demonstrate each phase of the installation process.

---

## Prerequisites

Before beginning the installation process, ensure the following prerequisites are met:

- A computer running macOS, Windows, or Linux operating system
- Active internet connection for downloading the installer
- Administrator or sudo privileges for installing software
- Sufficient disk space (minimum 500 MB recommended)
- Web browser installed and accessible

---

## Task 1: Downloading Cursor IDE

### Description

The first task involves downloading the Cursor IDE installer from the official website. This requires accessing the Cursor website, identifying the correct version for the operating system, and downloading the installation package.

### Procedure

1. **Open Web Browser**: Launch your preferred web browser (Chrome, Firefox, Safari, or Edge).

2. **Navigate to Cursor Website**: Type the following URL in the address bar:
   ```
   https://cursor.sh
   ```
   Press Enter to navigate to the official Cursor website.

3. **Locate Download Section**: On the homepage, look for the "Download" button or link. This is typically prominently displayed on the main page.

4. **Select Operating System**: Click on the download button. The website will automatically detect your operating system, or you may need to manually select:
   - **macOS**: Select macOS version (downloads `.dmg` file)
   - **Windows**: Select Windows version (downloads `.exe` installer)
   - **Linux**: Select Linux version (downloads `.AppImage`, `.deb`, or `.rpm` package)

5. **Initiate Download**: Click the download button for your operating system. The download will begin automatically.

6. **Monitor Download Progress**: Wait for the download to complete. The file will be saved to your default Downloads folder.

### Screenshot

**Figure 1.1:** Cursor website homepage showing the download section  
*Location: `screenshots/01_cursor_website_download.png`*

---

## Task 2: Installing Cursor IDE

### Description

This task involves installing the downloaded Cursor IDE package on the local system. The installation process varies depending on the operating system, but generally involves running the installer and following the installation wizard.

### Procedure for macOS

1. **Locate Downloaded File**: Open Finder and navigate to the Downloads folder. Locate the downloaded `.dmg` file named `Cursor-[version].dmg`.

2. **Open Disk Image**: Double-click the `.dmg` file to mount it. A new Finder window will open showing the Cursor application.

3. **Install Application**: Drag the Cursor icon to the Applications folder icon in the same window. This will copy Cursor to the Applications folder.

4. **Wait for Copy**: Wait for the copy operation to complete. A progress indicator will show the copying status.

5. **Eject Disk Image**: Once copying is complete, eject the disk image by clicking the eject button next to the mounted disk image, or right-click and select "Eject".

6. **Security Settings** (if prompted): On first launch, macOS may display a security warning. If this occurs:
   - Go to System Preferences → Security & Privacy
   - Click "Open Anyway" next to the Cursor message
   - Confirm the action

### Procedure for Windows

1. **Locate Downloaded File**: Open File Explorer and navigate to the Downloads folder. Locate the downloaded `.exe` file named `Cursor-Setup-[version].exe`.

2. **Run Installer**: Double-click the `.exe` file to launch the installer. If Windows displays a security warning, click "Run" or "Yes" to proceed.

3. **User Account Control**: If prompted by User Account Control (UAC), click "Yes" to allow the installer to make changes to your system.

4. **Follow Installation Wizard**:
   - **Welcome Screen**: Click "Next" to begin the installation
   - **License Agreement**: Read the license terms and select "I accept the agreement", then click "Next"
   - **Installation Location**: Choose the installation directory (default is recommended) and click "Next"
   - **Additional Options**: Select any additional options such as creating desktop shortcuts, then click "Next"
   - **Ready to Install**: Review the installation settings and click "Install"

5. **Wait for Installation**: The installer will copy files and set up Cursor. A progress bar will show the installation status.

6. **Complete Installation**: Once installation is complete, click "Finish" to exit the installer.

### Screenshot

**Figure 2.1:** Installation process (macOS: disk image with Cursor application / Windows: installer wizard)  
*Location: `screenshots/02_installation_process.png`*

**Note:** For macOS, this shows the mounted disk image with Cursor ready to be dragged to Applications. For Windows, this shows the installation wizard during setup.

---

## Task 3: First Launch and Initial Setup

### Description

After installation, the first launch of Cursor requires initial setup and configuration. This includes accepting terms of service, choosing preferences, and optionally signing in to enable cloud features.

### Procedure

1. **Launch Cursor**: 
   - **macOS**: Open Finder, go to Applications, and double-click Cursor
   - **Windows**: Click the Start menu, search for "Cursor", and click the application
   - **Linux**: Run Cursor from the applications menu or terminal

2. **Welcome Screen**: On first launch, Cursor will display a welcome screen with options to:
   - Get started with Cursor
   - Sign in to Cursor account (optional)
   - Skip and continue

3. **Accept Terms**: Read and accept the Terms of Service and Privacy Policy if prompted.

4. **Choose Theme**: Select your preferred color theme:
   - **Dark Theme**: Better for low-light environments
   - **Light Theme**: Better for bright environments
   - You can change this later in settings

5. **Sign In (Optional)**: If you wish to use cloud features, you can sign in using:
   - Email address
   - GitHub account
   - Google account
   - Or skip this step for now

6. **Initial Interface**: After setup, Cursor will open to the main editor interface showing the welcome page or file explorer.

### Screenshot

**Figure 3.1:** Main Cursor interface after first launch and initial setup  
*Location: `screenshots/03_cursor_main_interface.png`*

---

## Task 4: Configuring Cursor Settings

### Description

This task involves configuring Cursor's settings to optimize the development environment according to personal preferences and course requirements. Settings include editor preferences, font configuration, and extension installation.

### Procedure

1. **Open Settings**:
   - Press `Cmd + ,` (macOS) or `Ctrl + ,` (Windows/Linux)
   - Or go to: Cursor → Preferences → Settings (macOS) or File → Preferences → Settings (Windows/Linux)

2. **Configure Editor Settings**:
   - **Font Size**: Set to a comfortable size (recommended: 12-14)
   - **Font Family**: Choose a monospace font (e.g., 'Fira Code', 'Consolas', 'Monaco')
   - **Tab Size**: Set to 4 spaces (recommended for Python)
   - **Word Wrap**: Enable for better code readability
   - **Line Numbers**: Enable to show line numbers
   - **Minimap**: Enable/disable based on preference

3. **Configure File Settings**:
   - **Auto Save**: Enable "afterDelay" to auto-save files
   - **Default Line Ending**: Set to LF (Unix) or CRLF (Windows) based on OS
   - **Trim Trailing Whitespace**: Enable to remove trailing spaces

4. **Install Recommended Extensions**:
   - Click on the Extensions icon in the sidebar (or press `Cmd+Shift+X` / `Ctrl+Shift+X`)
   - Search for and install:
     - **Python** (by Microsoft) - For Python development
     - **Jupyter** (by Microsoft) - For Jupyter notebook support
     - **GitLens** (optional) - Enhanced Git capabilities
     - **Pylance** (by Microsoft) - Python language server

5. **Configure Python Interpreter** (if Python is installed):
   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the Python interpreter from the list

6. **Save Settings**: Settings are automatically saved. Close the settings tab when done.

### Screenshot

**Figure 4.1:** Cursor settings and extensions configuration  
*Location: `screenshots/04_settings_extensions.png`*

---

## Task 5: Verifying Installation

### Description

The final task involves verifying that Cursor IDE has been successfully installed and is functioning correctly. This includes checking the version, testing basic features, and confirming that all components are working properly.

### Procedure

1. **Check Cursor Version**:
   - Go to: Cursor → About Cursor (macOS) or Help → About (Windows/Linux)
   - Note the version number displayed
   - Verify it matches the latest version from the website

2. **Test File Operations**:
   - Create a new file: File → New File (or `Cmd+N` / `Ctrl+N`)
   - Save the file: File → Save (or `Cmd+S` / `Ctrl+S`)
   - Verify the file is saved correctly

3. **Test AI Features**:
   - Open the AI chat: Press `Cmd+L` (macOS) or `Ctrl+L` (Windows/Linux)
   - Type a simple query to test the AI functionality
   - Verify the AI responds appropriately

4. **Test Code Editing**:
   - Create a simple Python file (e.g., `test.py`)
   - Type some Python code
   - Verify syntax highlighting works
   - Test code completion (IntelliSense)

5. **Test Terminal Integration**:
   - Open integrated terminal: View → Terminal (or `` Ctrl+` ``)
   - Verify terminal opens and functions correctly
   - Test running a simple command (e.g., `python --version`)

6. **Verify Extensions**:
   - Go to Extensions view
   - Confirm all installed extensions are active and enabled

### Screenshot

**Figure 5.1:** About Cursor showing version information and verification  
*Location: `screenshots/05_about_cursor_verification.png`*

---

## Results and Observations

### Installation Summary

- ✅ Cursor IDE successfully downloaded from official website
- ✅ Installation completed without errors
- ✅ Application launches correctly
- ✅ Initial setup completed
- ✅ Settings configured according to preferences
- ✅ Required extensions installed and active
- ✅ All features tested and working properly

### Version Information

- **Cursor Version**: [Version Number - to be filled after installation]
- **Installation Date**: [Date]
- **Operating System**: [Your OS and Version]

### Observations

During the installation process, the following observations were made:

1. **Download Process**: The download was straightforward, and the website automatically detected the operating system, making it easy to select the correct installer.

2. **Installation Speed**: The installation completed quickly, taking approximately [X] minutes depending on system performance.

3. **User Interface**: Cursor's interface is clean and intuitive, similar to VS Code but with enhanced AI features readily accessible.

4. **First Launch**: The welcome screen provided clear guidance for initial setup, making it easy for new users to get started.

5. **Settings Configuration**: The settings interface is well-organized, allowing easy customization of the editor according to personal preferences.

6. **Extension Installation**: The extension marketplace is well-integrated, and installing extensions is a seamless process.

7. **AI Features**: The AI chat feature (Cmd/Ctrl + L) is easily accessible and provides helpful assistance for coding tasks.

---

## Conclusion

[**IMPORTANT: Write your own conclusion here. Do not copy-paste or use AI-generated content. The conclusion should reflect your personal experience and what you learned from completing this lab.**]

Through completing this lab, I learned about the process of installing and configuring a modern IDE. The installation of Cursor IDE was straightforward, and I gained hands-on experience with setting up a development environment. I observed how the installation process varies slightly between different operating systems, particularly in how macOS uses disk images while Windows uses executable installers.

The configuration phase taught me the importance of customizing development tools to match personal preferences and workflow requirements. I learned how to navigate settings menus, install extensions, and configure the editor for optimal productivity. The verification process helped me understand how to systematically test software installation to ensure all components are functioning correctly.

This lab provided valuable experience in setting up development tools, which is an essential skill for any programmer. Understanding the installation and configuration process will be beneficial for future labs and projects in this course.

---

## References

1. Cursor Official Website. (n.d.). Retrieved from https://cursor.sh
2. Cursor Documentation. (n.d.). Retrieved from https://docs.cursor.sh
3. Visual Studio Code Documentation. (n.d.). Retrieved from https://code.visualstudio.com/docs

---

## Appendix: Screenshots

### Screenshot Index

All screenshots referenced in this report are stored in the `screenshots` folder within the Lab 1 directory. The following is a complete list of all screenshots:

1. **01_cursor_website_download.png** - Cursor website homepage showing download section (can be found online at cursor.sh)
2. **02_installation_process.png** - Installation process (macOS: disk image / Windows: installer wizard)
3. **03_cursor_main_interface.png** - Main Cursor interface after first launch and initial setup
4. **04_settings_extensions.png** - Cursor settings and extensions configuration
5. **05_about_cursor_verification.png** - About Cursor showing version information and verification

### Screenshot Guidelines

- All screenshots should be clear and readable
- Screenshots should capture the relevant portion of the screen
- File names should match exactly as listed above
- Screenshots should be in PNG format for best quality
- Ensure screenshots are properly cropped to focus on relevant content

---

**End of Lab Report 1**

**Prepared by:** [Your Full Name]  
**Roll Number:** [Your Roll Number]  
**Section:** [Your Section]  
**Date:** [Date]
