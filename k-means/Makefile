all: tbb pthread victor_mp

tbb: tbb.cpp
	g++ tbb.cpp -std=c++11 -O2 -ltbb -o tbb

pthread: pthread.cpp
	g++ pthread.cpp -std=c++11 -pthread -o pthread -O2

victor_mp: immint.cpp
	g++ immint.cpp -std=c++11 -o immint -g -fopenmp -O2 -march=native
