CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lzmq -lczmq -lnanomsg -lnng
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)
EXEC = bml-pub-sub

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)
	rm -rf ../logs logs
	rm -rf 127.0.0.1* benchmark* > /dev/null 2>&1

clean_results_data:
	rm -rf data

clean_all: clean clean_results_data

profiling: CFLAGS += -pg -ltcmalloc
profiling: $(EXEC)