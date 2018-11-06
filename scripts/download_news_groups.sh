#!/usr/bin/env bash

USAGE="$0 [-t <path to target directory>]"

# default target directory
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. >/dev/null && pwd )"
TARGET_DIR=${BASEDIR}/datasets

while getopts ':t:' opt
do
    case $opt in
         t) TARGET_DIR=$OPTARG;;
        \?) echo "ERROR: Invalid option: $USAGE" exit 1;;
    esac
done

NEWSGROUPS_DIR=${TARGET_DIR}/NewsGroups

mkdir -p ${NEWSGROUPS_DIR}/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz
tar -xf 20_newsgroups.tar.gz -C ${NEWSGROUPS_DIR} && rm 20_newsgroups.tar.gz
tar -xf mini_newsgroups.tar.gz -C ${NEWSGROUPS_DIR} && rm mini_newsgroups.tar.gz
