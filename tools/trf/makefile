
DIR_BASE=./src/base/
DIR_TRF=./src/TRF/
CC=g++ -I $(DIR_BASE) -std=c++11 -fopenmp -m64 -O3

CPP=$(DIR_BASE)wb-log.cpp $(DIR_BASE)wb-linux.cpp $(DIR_BASE)wb-win.cpp $(DIR_BASE)wb-file.cpp \
		$(DIR_BASE)wb-option.cpp $(DIR_BASE)wb-solve.cpp $(DIR_BASE)wb-string.cpp \
		$(DIR_TRF)trf-def.cpp $(DIR_TRF)trf-vocab.cpp $(DIR_TRF)trf-alg.cpp $(DIR_TRF)trf-corpus.cpp \
		$(DIR_TRF)trf-feature.cpp $(DIR_TRF)trf-model.cpp \
		$(DIR_TRF)trf-ml-train.cpp $(DIR_TRF)trf-sa-train.cpp 
	
OBJ=$(CPP:.cpp=.o)

TMH=$(DIR_BASE)wb-vector.h $(DIR_BASE)wb-mat.h $(DIR_BASE)wb-heap.h $(DIR_BASE)wb-iter.h $(DIR_BASE)wb-lhash.h \
		$(DIR_BASE)wb-trie.h $(DIR_BASE)wb-system.h 
		 
MAIN_TRAIN=$(DIR_TRF)main-SA-train.cpp $(DIR_TRF)main-ML-train.cpp
MAIN_TRF=$(DIR_TRF)main-TRF.cpp	 

DIR_EXE=./bin/
EXE_SA=$(DIR_EXE)trf_satrain
EXE_ML=$(DIR_EXE)trf_mltrain
EXE_TRF=$(DIR_EXE)trf


all: bulid $(EXE_ML) $(EXE_SA) $(EXE_TRF)

bulid: 
	mkdir -p $(DIR_EXE)

$(EXE_ML): $(OBJ) $(TMH) $(MAIN_TRAIN)
	echo ---> $@ 
	$(CC) $(OBJ) $(TMH) $(MAIN_TRAIN) -o $@ -D _MLTrain
	
$(EXE_SA): $(OBJ) $(TMH) $(MAIN_TRAIN)
	echo ---> $@ 
	$(CC) $(OBJ) $(TMH) $(MAIN_TRAIN) -o $@ 

$(EXE_TRF): $(OBJ) $(TMH) $(MAIN_TRF)
	echo ---> $@ 
	$(CC) $(OBJ) $(TMH) $(MAIN_TRF) -o $@ 
	
%.o: %.cpp %.h
	$(CC) -c $< -o $@
	
clean:
	rm -f $(OBJ)
	rm -f $(EXE_ML) $(EXE_SA) $(EXE_TRF)