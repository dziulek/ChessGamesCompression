CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -L. -lfast_pgn_to_uci -lm 
TARGET = program

SOURCES = main.c fast_pgn_to_uci.c
HEADERS = fast_pgn_to_uci.h
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) libfast_pgn_to_uci.a
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< -lm

libfast_pgn_to_uci.a: fast_pgn_to_uci.o
	ar rcs $@ $^

clean:
	rm -f $(OBJECTS) $(TARGET) lfast_pgn_to_uci.a
