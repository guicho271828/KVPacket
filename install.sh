#!/bin/bash -xe

ENV=kvpacket

CONDA=""
if which mamba > /dev/null
then
    CONDA=$(which mamba)
elif which conda > /dev/null
then
    CONDA=$(which conda)
fi

if [ -z $CONDA ]
then
    echo "Error: conda or mamba is not installed or is not in the PATH."
    echo "Go to "
    echo "* https://github.com/conda-forge/miniforge (open source)"
    echo "* https://www.anaconda.com/download/success (registration required)"
    echo "to obtain a conda/mamba installer."
else
    echo "Using $CONDA for environment setup"
fi


BACKEND=""
usage(){
    echo "Usage: install.sh [-h] [-y] [-b vllm|sglang]"
    echo
    echo "-h : show this help"
    echo "-y : Adds '-y' option to '$CONDA env [create|remove|...]' command arguments."
    echo "-b : Backend to install (vllm or sglang). Optional. If omitted,"
    echo "     only the huggingface backend (always in core) is available."
    echo "     You can run 'askllm select <backend>' later to add one."
    exit 1
}

CONDA_OPTIONS=""
while getopts "yhb:" OPTNAME ; do
    case "${OPTNAME}" in
        h)
            usage
            ;;
        y)
            CONDA_OPTIONS="-y"
            ;;
        b)
            if [ "${OPTARG}" != "vllm" ] && [ "${OPTARG}" != "sglang" ]; then
                echo "Error: -b must be one of: vllm, sglang"
                usage
            fi
            BACKEND="${OPTARG}"
            ;;
        :)
            # If expected argument omitted:
            echo "Error: -${OPTARG} requires an argument."
            exit 1
            ;;
        *)
            # If unknown (any other) option:
            echo "Error: -${OPTARG} unknown."
            exit 1
            ;;
    esac
done

if $CONDA env list | grep -q $ENV
then
    echo "An existing $ENV environment was found."
    $CONDA env remove $CONDA_OPTIONS -n $ENV
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
$CONDA env create $CONDA_OPTIONS -f $SCRIPT_DIR/environment.yml

$CONDA run -n $ENV uv pip install -e . --group dev

# # these libraries are NOT necessary for *building* this library.
# # they are merely "commonly necessary" for *using* transformer effectively.
# # Therefore they should not be added as dependency.
# $CONDA run -n $ENV uv pip install accelerate compressed-tensors protobuf sentencepiece datasets

# If a backend was requested, activate it: installs the [vllm] or [sglang]
# extra deps and writes askllm/_active_backend.json so `import askllm` knows
# which backend to re-export. Without -b, the env has only the huggingface
# backend; the user can run `askllm select <backend>` later to add one.
if [ -n "$BACKEND" ]; then
    $CONDA run -n $ENV askllm select $BACKEND
fi

# if [ "$BACKEND" = "vllm" ]; then
#     # vllm pulls a transformers version that conflicts with autoawq,
#     # so autoawq must be installed after vllm.
#     $CONDA run -n $ENV uv pip install autoawq
# fi

# $CONDA run -n $ENV uv pip install pre-commit
# $CONDA run -n $ENV pre-commit install
