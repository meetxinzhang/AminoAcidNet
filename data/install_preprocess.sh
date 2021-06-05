git clone https://github.com/gjoni/mylddt
mv mylddt ./data/preprocess
# shellcheck disable=SC2164
cd data/preprocess/src/
g++ -Wall -Wno-unused-result -pedantic -O3 -mtune=native -std=c++11 *.cpp -o get_features
mv get_features ../


