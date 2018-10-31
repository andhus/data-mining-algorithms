#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DATASET_DIR=${DIR}/NewsGroups

mkdir -p $DATASET_DIR
wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz
tar -xf 20_newsgroups.tar.gz -C $DATASET_DIR && rm 20_newsgroups.tar.gz
tar -xf mini_newsgroups.tar.gz -C $DATASET_DIR && rm mini_newsgroups.tar.gz
