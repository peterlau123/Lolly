#!/bin/bash
src_files=`find include/ source/ test/ -regextype posix-extended -regex ".*\.(c|h|cc|hh|cpp|hpp|cu)"`
for file in ${src_files}
do
        echo ${file}
        clang-format -i ${file}
done
echo "Buddy,well done!"
