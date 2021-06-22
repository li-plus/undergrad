#!/bin/bash

if [ "$1" = "" ]; then
	echo Usage: ./run.sh testcase
	exit 1
fi

exec ../../PA4_build/test/unit_tests --dataset $1 --datadir ../data/ --len 32

