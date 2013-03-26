SOURCES = src/regression.rkt
OUTPUT = regression

all: $(SOURCES)
	racket $^

exe: $(SOURCES)
	raco exe -o $(OUTPUT) $^
