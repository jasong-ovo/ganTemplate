#!/bin/bash
# used in data/path

fid_path="./mnist/raw/fid"
datas_path="./mnist/raw/train"
N=2049
cnt=1

loop_path=$datas_path"/*"
if [ ! -d $fid_path ]; then

  mkdir $fid_path
  for file in $loop_path
  do
    echo "$file"
    if [ $cnt -gt $N ]
    then
        break
    fi
    cp $file $fid_path
    cnt=$((cnt+1))
  done

else
  echo "$fid_path exists"
fi
