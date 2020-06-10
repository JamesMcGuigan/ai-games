#!/usr/bin/env bash
# Utility script to automate python virtualenv creation/update from ./requirements.in
# Input:  ./requirements.in
# Output: ./requirements.txt + ./venv/ + ./venv_windows/
#
# Redownload latest version of script:
# wget https://raw.githubusercontent.com/JamesMcGuigan/requirements.sh/master/requirements.sh -O requirements.sh

set -x
cd $(dirname $(readlink -f ${BASH_SOURCE[0]}));  # OSX requires: brew install coreutils

PYTHON_VERSION=3

for OS in UNIX WINDOWS; do
    if [[ $OS == 'UNIX' ]]; then
        if [[ $PYTHON_VERSION == 2 ]]; then
            # OSX python doesn't have openssl support for pip: brew install python@2
            # if [[ `which brew` ]]; then brew install python@2; fi;

            PYTHON=`     which python2`
            PIP=`        which pip2`
            VIRTUALENV="`which virtualenv` -p $PYTHON"
        fi
        if [[ $PYTHON_VERSION == 3 ]]; then
            # Ensure Ubuntu dependencies are installed
            if [[ `command -v dpkg` ]]; then for PACKAGE in python3 python3-pip python3-venv; do
              if [[ ! `dpkg -l | grep -w $PACKAGE` ]]; then sudo apt install $PACKAGE; fi;
            done; fi;

            PYTHON=`which python3`
            PIP=`   which pip3`
            VIRTUALENV="$PYTHON -m venv"
        fi
        VENV=venv
        VENV_BIN=./$VENV/bin
        VENV_ACTIVATE=./$VENV/bin/activate
    fi;
    if [[ $OS == 'WINDOWS' ]]; then
        if [[ $PYTHON_VERSION == 2 ]]; then
            PYTHON=`     find /c/Users/$USER/AppData/Local/Programs/Python/Python2* -name 'python.exe'     | sort -nr | head -n 1`
            PIP=`        find /c/Users/$USER/AppData/Local/Programs/Python/Python2* -name 'pip.exe'        | sort -nr | head -n 1`
            VIRTUALENV=` find /c/Users/$USER/AppData/Local/Programs/Python/Python2* -name 'virtualenv.exe' | sort -nr | head -n 1`  # untested on Windows
        fi
        if [[ $PYTHON_VERSION == 3 ]]; then
            PYTHON=`find /c/Users/$USER/AppData/Local/Programs/Python/Python3* -name 'python.exe' | sort -nr | head -n 1`
            PIP=`   find /c/Users/$USER/AppData/Local/Programs/Python/Python3* -name 'pip.exe'    | sort -nr | head -n 1`
            VIRTUALENV="$PYTHON -m venv"
        fi
        VENV=venv_windows
        VENV_BIN=./$VENV/Scripts
        VENV_ACTIVATE=./$VENV/Scripts/activate

        if [[ ! $PYTHON ]]; then continue; fi;  # don't install ./venv_windows/ if not on Windows
    fi

    # Install venv and dependencies
    if [[ ! -d ./$VENV/ ]]; then
        $VIRTUALENV ./$VENV/;

        if [[ $OS == 'WINDOWS' ]]; then
            dos2unix $VENV_ACTIVATE
            perl -p -i -e "s/\"(C:.*)\"/'\$1'/g"       $VENV_ACTIVATE  # BUGFIX: syntax error near unexpected token `(' if [ "x(venv) " != x ] ; then'
        fi
        echo "export PATH=$VENV_BIN:\$PATH"                                      >> $VENV_ACTIVATE
        echo "export PYTHONSTARTUP=./.pythonstartup.py"                          >> $VENV_ACTIVATE
        echo "export PYTHONPATH='`pwd`'"                                         >> $VENV_ACTIVATE
        echo "PS1=\"($VENV) \$(echo \$PS1 | perl -p -e 's/^(\s*\(.*?\))+//g')\"" >> $VENV_ACTIVATE  # BUGFIX: $PS1 prompt
    fi;

    # pip-compile without argument requires ./requirements.txt to exist
    if [[ ! -f ./requirements.in   ]]; then touch ./requirements.in;   fi;
    if [[ ! -f ./requirements.txt  ]]; then touch ./requirements.txt;  fi;
    if [[ ! -f ./.pythonstartup.py ]]; then touch ./.pythonstartup.py; fi;

    # Use pip and python from inside the virtualenv
    source $VENV_ACTIVATE
    if [[ $OS == 'UNIX' ]]; then
        pip install --upgrade pip pip-tools
        timeout 5 pip-compile || pip-compile -v  # --generate-hashes
        pip install -r ./requirements.txt || cat ./requirements.txt | perl -p -e 's/\s*#.*$//g' | sed '/^\s*$/d' | xargs -d'\n' -L1 -t pip install
        pip-sync
    fi;
    if [[ $OS == 'WINDOWS' ]]; then
        python -m pip install --upgrade pip pip-tools
        pip-compile.exe
        python -m pip install -r ./requirements.txt
        pip-sync.exe
    fi;
done
