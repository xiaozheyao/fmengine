#!/bin/bash

echo "Running compilance check"

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

pushd $PARENT_DIR

echo "Running auto-formatters"

isort . > /dev/null
autopep8 fmengine examples --recursive --in-place --pep8-passes 2000 > /dev/null
black fmengine examples --verbose --config black.toml > /dev/null

echo "Running linters"

if flake8 --max-line-length 120 fmengine ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if black --check fmengine --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if pylint fmengine ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

echo "Running tests"

if pytest ; then
    echo "No pytest errors"
else
    echo "pytest errors"
    exit 1
fi

if mypy fmengine ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successful compliance check"

popd