#!/bin/bash

dsets=(arxiv collab citation ddi protein ppa reddit.dgl products youtube amazon_cogdl yelp wikikg2 am)
filename=output_$(date +"%H_%M_%S_%m_%d").log

echo Log saved to $filename

for j in `seq 0 $((${#dsets[@]}-1))`;
do
    echo ${dsets[j]}
    ../../PA4_build/test/unit_tests --dataset ${dsets[j]} --datadir ../data/  --len 32  2>&1 | tee -a $filename 
done

