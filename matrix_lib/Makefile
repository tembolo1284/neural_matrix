BINARY=main
LIBRARY=libmatrix
LIBDIR=./library
TESTCOVERAGEDIR=./test_coverage

TEST=tests
TESTS=$(wildcard $(TEST)/*.c)
TESTBINS=$(patsubst $(TEST)/%.c, $(TEST)/bin/%, $(TESTS))

CODEDIRS=. src
INCDIRS=. ./include/

CC=gcc
OPT=-O0

# generate files that encode make rules for the .h dependencies
DEPFLAGS=-MP -MD

# automatically add the -I onto each include directory
CFLAGS=-Wall -Wextra -Werror -Wpedantic -g $(foreach D,$(INCDIRS),-I$(D)) $(OPT) $(DEPFLAGS) -fPIC -fprofile-arcs -ftest-coverage

# for-style iteration (foreach) and regular expression completions (wildcard)
CFILES=$(foreach D,$(CODEDIRS),$(wildcard $(D)/*.c))

# regular expression replacement
OBJECTS=$(patsubst %.c,%.o,$(filter-out $(TESTS), $(CFILES)))
DEPFILES=$(patsubst %.c,%.d,$(CFILES))

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $@ $^ -lgcov -lm

# only want the .c file dependency here, thus $< instead of $^.

%.o:%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BINARY) $(OBJECTS) $(DEPFILES) $(TESTBINS) $(LIBDIR) $(TESTCOVERAGEDIR) *.gcda *.gcno *.gcov coverage.info
	rm -rf tests/bin/
	rm -rf src/*.gcda src/*.gcno

$(LIBRARY).a: $(OBJECTS)
	ar rcs $@ $^

$(LIBRARY).so: $(OBJECTS)
	$(CC) -shared -o $@ $^

# shell commands are a set of keystrokes away
distribute: clean $(LIBRARY).a $(LIBRARY).so
	mkdir -p $(LIBDIR)
	mv $(LIBRARY).a $(LIBRARY).so $(LIBDIR)
	tar zcvf dist.tgz *

# @ silences the printing of the command
# $(info ...) prints output
diff:
	$(info The status of the repository, and the volume of per-file changes:)
	@git status
	@git diff --stat

$(TEST)/bin/%: $(TEST)/%.c src/matrix.c
	mkdir -p $(TEST)/bin
	$(CC) $(CFLAGS) -o $@ $^ -lcriterion --coverage -lm

test: $(TESTBINS)
	for test in $(TESTBINS) ; do ./$$test ; done

# New target for coverage
test_coverage:
	mkdir -p test_coverage
	lcov --capture --directory . --output-file test_coverage/coverage.info
	lcov --remove test_coverage/coverage.info '/usr/*' --output-file test_coverage/coverage.info
	genhtml test_coverage/coverage.info --output-directory test_coverage/report

# include the dependencies
-include $(DEPFILES)

# add .PHONY so that the non-targetfile - rules work even if a file with the same name exists.
.PHONY: all clean distribute diff test test_coverage
