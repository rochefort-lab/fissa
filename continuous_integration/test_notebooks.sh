#!/usr/bin/env bash

# test_notebooks
# Turns jupyter notebooks (.ipynb) into python scripts and tries to run them
#
# Inputs
#   - (Optional) -i flag, which operates in place instead of in a temporary dir
#   - (Optional) -v flag, which increases verbosity
#   - (Optional) -q flag, which suppresses all output except in the event
#     of failure.
#   - A path to either a notebook or a directory containing jupyter notebooks
#
# Outputs
#   - Progress reports (can be suppressed with -q flag)
#   - Output from called scripts can be enabled with -v flag.
#
# Exit codes
#   - 0 = all notebooks ran to completion
#   - else, the number of notebooks which aborted when run as a script

# =============================================================================

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

[ $# -eq 0 ] && usage

IN_PLACE_MODE="false";
VERBOSITY=0;

XC='\033[0;31m'  # Red
NC='\033[0m'  # No colour


while getopts "hiqv" arg; do
  case $arg in
    i) # Enable in-place mode
      IN_PLACE_MODE="true";
      ;;
    q) # Decrease verbosity
      VERBOSITY=$((VERBOSITY - 1));
      ;;
    v) # Increase verbosity
      VERBOSITY=$((VERBOSITY + 1));
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))
# now do something with $@
SOURCE_PATH="$1";

if [ "$VERBOSITY" -ge 1 ]; then
    echo "[TESTNB] Source path provided is '$SOURCE_PATH'";
fi

if [[ -z "${SOURCE_PATH// }" ]]; then
    echo "[TESTNB] Source path can not be an empty string.";
    exit -1;
fi

if [ ! -e "$SOURCE_PATH" ]; then
    echo "[TESTNB] Source path input does not exist.";
    exit -1;
fi

if [ "$IN_PLACE_MODE" == "true" ]; then
    tmp_folder="";
else
    # Use the process name as a subfolder, so we can run multiple instances
    # at once without them overwriting each other
    tmp_folder="/tmp/test_notebooks/$$";
    mkdir -p "$tmp_folder";
fi

if [ -d "$SOURCE_PATH" ]; then
    input_type="dir";
fi

if [ "$IN_PLACE_MODE" == "true" ]; then
    if [ -d "$SOURCE_PATH" ]; then
        cd "$SOURCE_PATH";
    else
        cd $(dirname "${SOURCE_PATH}");
    fi
else
    if [ -d "$SOURCE_PATH" ]; then
        SOURCE_PATH="${SOURCE_PATH%/}";
        cp -r "$SOURCE_PATH/"* "$tmp_folder";
    elif [ -f "$SOURCE_PATH" ]; then
        cp "$SOURCE_PATH" "$tmp_folder";
    else
        echo "[TESTNB] Source path input must be a directory or a file";
        exit -1;
    fi

    cd "$tmp_folder";
fi

if [ "$input_type" == 'dir' ]; then
    NB_FILE_LIST=(*.ipynb);
else
    NB_FILE_LIST=$(basename "$SOURCE_PATH");
fi

OIFS="$IFS"
IFS=$'\n'

RETURNVALUE=0;
for NBFILE in ${NB_FILE_LIST[@]};
do
    if [ "$VERBOSITY" -ge 0 ]; then
        echo "==================================================================";
        echo "[TESTNB] Converting notebook '$NBFILE' into a script";
        echo "------------------------------------------------------------------";
    fi;
    CONVERT_OUTPUT=$(jupyter nbconvert --to=script "$NBFILE");
    CONVERT_STATUS=$?;
    if [ $CONVERT_STATUS -ne 0 ]; then
        RETURNVALUE=$((RETURNVALUE + 1));
        echo -e "${XC}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        echo -e "${XC}[TESTNB] Notebook '$NBFILE' could not be converted into a script";
        echo -e "${XC}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}";
#        echo "$CONVERT_OUTPUT";
        continue;
    fi
#    if [ "$VERBOSITY" -ge 2 ]; then
#        echo "$CONVERT_OUTPUT";
#    fi
    PYFILE="${NBFILE%.*}.py";
    # Comment out any lines in $PYFILE which are calls to help()
    sed -ie 's/^\(help(.*)\)/#\ \1/' "$PYFILE";
    if [ "$VERBOSITY" -ge -1 ]; then
        echo "------------------------------------------------------------------";
        echo "[TESTNB] Trying to run the generated script '$PYFILE'";
        echo "------------------------------------------------------------------";
    fi
    NB_OUTPUT=$(ipython "$PYFILE")
    NB_STATUS=$?;
    if [ $NB_STATUS -ne 0 ]; then
        RETURNVALUE=$((RETURNVALUE + 1));
        echo -e "${XC}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        echo -e "${XC}[TESTNB] Script '$NBFILE' exited with status=$NB_STATUS";
        echo -e "${XC}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}";
    fi
    if [[ "$VERBOSITY" -ge 1 || $NB_STATUS -ne 0 ]]; then
        echo "$NB_OUTPUT";
    fi
    if [ "$VERBOSITY" -ge 0 ]; then
        echo "------------------------------------------------------------------";
        echo "[TESTNB] Script '$NBFILE' exited with status=$NB_STATUS";
    fi
    if [ "$VERBOSITY" -ge 2 ]; then
        echo "[TESTNB] Current exit code is $RETURNVALUE";
    fi
done;

IFS="$OIFS"

#------------------------------------------------------------------------------
# Tear down

if [ "$IN_PLACE_MODE" != "true" ]; then
    if [ "$VERBOSITY" -ge 0 ]; then
        echo "==================================================================";
        echo "[TESTNB] Removing temporary files";
    fi
    rm -rf "$tmp_folder";
fi
if [ "$VERBOSITY" -ge 0 ]; then
    echo "==================================================================";
    echo "[TESTNB] Done";
fi

#------------------------------------------------------------------------------
# Exit

exit $RETURNVALUE;
