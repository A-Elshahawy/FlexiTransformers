#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Sound configuration 
# ERROR_SOUND="/usr/share/sounds/freedesktop/stereo/dialog-error.oga"
# ERROR_SOUND="C:\Users\aelsh\Downloads\Music\wahwa.mp3"
function play_error_sound() {
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        powershell -c "(New-Object System.Media.SoundPlayer).Stream=(New-Object System.IO.MemoryStream (,[System.IO.File]::ReadAllBytes('$ERROR_SOUND'))); (New-Object System.Media.SoundPlayer).PlaySync()"
    else
        echo -e "\a"
    fi
}

function run_checks() {
 local errors=0
 
 echo "Running pre-commit checks..."
 
 # Run pre-commit
 if ! pre-commit run --all-files; then
   echo -e "${RED}❌ Pre-commit checks failed${NC}"
   errors=$((errors + 1))
 fi


 if [ $errors -gt 0 ]; then
   echo -e "${RED}Found $errors error(s)${NC}"
   play_error_sound
   exit 1
 else
   echo -e "${GREEN}✓ All checks passed${NC}"
   exit 0
 fi
}

run_checks