#!/bin/bash
if [ $# -lt 1 ]
then
	echo "Usage: sh push_code.sh comments"
	echo $1 
	exit 0
fi

git add .
git add -u .
git commit -m "$1"
git push -u origin master
