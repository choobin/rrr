CXX = g++ -std=c++11 -march=native

CFLAGS = -Wall -Wextra -Werror -Wshadow -Wcast-qual -Wcast-align -Wwrite-strings -O2

CPPFLAGS = -DNDEBUG

INCLUDES =

LDFLAGS = -lm

DEBUG = #-ggdb

TARGETS = test-rrr

all: $(TARGETS)

test-rrr: test-rrr.o rrr.o
	$(CXX) $(DEBUG) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(DEBUG) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $<

clean:
	-$(RM) -f *.o $(TARGETS) >/dev/null 2>&1

remake:
	$(MAKE) clean && clear && $(MAKE) all
