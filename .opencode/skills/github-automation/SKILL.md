---
name: github-automation
description: Instructions for GitHub CLI and Git version control automation in OpenCode.
---

# GitHub Automation — OpenCode Skill

## Rules for Git & GitHub Operations

1. **GitHub CLI Path**:
   On Windows environments where `gh` is in `C:\Program Files\GitHub CLI\gh.exe`, invoke it via:
   `& 'C:\Program Files\GitHub CLI\gh.exe'` in PowerShell.

2. **Repository Creation**:
   To create a new repository under user account `pnt2005`:
   ```powershell
   & 'C:\Program Files\GitHub CLI\gh.exe' repo create pnt2005/CheXSpark --public --source=D:\BigData\CheXSpark --remote=origin --push
   ```

3. **Commit Conventions**:
   Use standard conventional commits:
   - `feat:` for new UI pages, components, API client features
   - `fix:` for bug fixes and layout adjustments
   - `docs:` for documentation, SKILL.md, and README updates
   - `chore:` for config and package setup

4. **Workflow Execution**:
   - Initialize git: `git init`
   - Set initial branch: `git branch -M main`
   - Add all files: `git add .`
   - Commit: `git commit -m "feat: initial commit"`
   - Push: `git push -u origin main`
