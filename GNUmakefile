SOURCES = src/regression.rkt
OUTPUT = regression

all: $(SOURCES)
	racket $^

run: all

exe: $(SOURCES)
	raco exe -o $(OUTPUT) $^
