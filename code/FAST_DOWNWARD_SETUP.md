# Fast Downward Setup Instructions

This project uses Fast Downward, a classical planning system. The build artifacts are not committed to the repository to keep it clean.

## Installation

### Option 1: Download and Build (Recommended)

1. Download Fast Downward 24.06.1:
   ```bash
   cd code/
   wget https://github.com/aibasel/downward/releases/download/release-24.06/fast-downward-24.06.1.tar.gz
   tar -xzf fast-downward-24.06.1.tar.gz
   rm fast-downward-24.06.1.tar.gz
   ```

2. Build Fast Downward:
   ```bash
   cd fast-downward-24.06.1
   python3 build.py release
   ```

### Option 2: Use System Package Manager

#### Ubuntu/Debian:
```bash
sudo apt-get install fast-downward
```

#### macOS with Homebrew:
```bash
brew install fast-downward
```

## Verification

Test that Fast Downward is working:
```bash
cd code/fast-downward-24.06.1
./fast-downward.py --help
```

## Project Integration

The project expects Fast Downward to be located at:
```
code/fast-downward-24.06.1/
```

If you install it elsewhere, update the path in `fast_downward_planner.py`.

## Notes

- Build artifacts (`builds/` directory) are excluded via `.gitignore`
- Source code and Python driver are included in the repository
- The planner requires Python 3.6 or newer
