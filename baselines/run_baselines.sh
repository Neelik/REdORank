#!/bin/bash

# This is a script to execute the RankLib models used as baselines in my thesis experiments.
# Note that for the -a option, a value must be passed with the flag, but the value does not matter. It will be ignored.
# Additionally, -m and -a can be entered simultaneously but the -m flag will be ignored in preference to -a.

DATASETDIR="/home/gallen/Projects/REdORank/datasets/"
TRAINFILE="mslr/Fold1/train.txt"
TESTFILE="mslr/Fold1/test.txt"
VALFILE="mslr/Fold1/vali.txt"

function LambdaMart {
  echo "Running LambdaMART";
  java -jar RankLib-2.16.jar -train "$DATASETDIR$TRAINFILE" -ranker 6 -metric2t NDCG@10 -test "$DATASETDIR$TESTFILE" -validate "$DATASETDIR$VALFILE"
}

function ListNet {
  echo "Running ListNET";
  java -jar RankLib-2.16.jar -train "$DATASETDIR$TRAINFILE" -ranker 7 -metric2t NDCG@10 -test "$DATASETDIR$TESTFILE" -validate "$DATASETDIR$VALFILE"
}

function AdaRank {
  echo "Running AdaRank";
  java -jar RankLib-2.16.jar -train "$DATASETDIR$TRAINFILE" -ranker 3 -metric2t NDCG@10 -test "$DATASETDIR$TESTFILE" -validate "$DATASETDIR$VALFILE"
}

function run {
  if [ "$all" = true ]
  then
    LambdaMart
    ListNet
    AdaRank
  else
    if [ "$model" = 1 ]
    then
      LambdaMart
    elif [ "$model" = 2 ]
    then
      ListNet
    elif [ "$model" = 3 ]
    then
      AdaRank
    fi
  fi
}

while getopts m:a: o
  do
    case "$o" in
      m) model=${OPTARG};;
      a) all=true;;
      [?]) print >"$2" "Usage: $0 [[-m 1|2|3] | [-a value]]"
        exit 1;;
      *) print >"$2" "Usage: $0 [[-m 1|2|3] | [-a value]]"
        exit 1;;
    esac
  done

if [ -a RankLib-2.16.jar ]
then
  run
else
  echo "Missing RankLib jar file."
fi