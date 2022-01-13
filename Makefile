#=====================================================================
# Build, test and install the geisten neural network library
#
# Typing 'make help' will print the available make commands
#
#======================================================================

PROJECT_NAME = geisten
PREFIX ?= /usr/local

MKDIR_P ?= mkdir -p
RM ?= rm

#--------------------------- DON'T change this part ----------------------------

SOURCE =
TESTS = test_geisten
DOCS = geisten.h

OBJ = $(SOURCE:.c=.o) $(addsuffix .o,$(TESTS))
DEP = $(OBJ:.o=.d)
DOCS_MD = $(DOCS:.h=.md)


# CFLAGS ?= -I. -march=native -mtune=native -MP -Wall -Wextra -mavx -Wstrict-overflow -ffast-math -fsanitize=address -O3 -MMD
CFLAGS ?= -I. -mtune=native -MP -Wall -Wextra -Wstrict-overflow  -ffast-math -O -MMD -g2
LDFLAGS ?= -ffast-math


options:
	@echo $(PROJECT_NAME) build options:
	@echo "CFLAGS   = ${CFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "CC       = ${CC}"

all: options test docs ## build all unit tests of the project

# compile the object files
%.o : %.c
	$(CC) $(CFLAGS) -c $< $(LIB_PATH) $(LIBS) -o $@ $(LDFLAGS)

# build the unit tests
test_%: test_%.o
	$(CC) -o $@ $< $(LDFLAGS)
	@./$@ ||  (echo "Test $^ failed" && exit 1)

test: $(TESTS) ## run all test programs
	@echo "Success, all tests of project '$(PROJECT_NAME)' passed."


.PHONY: clean
# clean the build
clean:  ## cleanup - remove the target (test) files
	rm -f $(OBJ) $(DEP) $(TESTS) $(DOCS_MD)

.PHONY: install
install: $(PROJECT_NAME).h  ## install the target build to the target directory ('$(DESTDIR)$(PREFIX)/include')
	install $< $(DESTDIR)$(PREFIX)/include/


# ----------------- TOOLS ------------------------------------------

# Create a markdown library documentation from a C header file
%.md: %.h
	@$(MKDIR_P) $(dir $@)
	./xtract.awk $< > $@

docs: $(DOCS_MD) ## build the documentation of the header files in markdown format

help: ## print this help information. Type 'make all' to build the project
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
		printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
		}' $(MAKEFILE_LIST)


# just type 'make print-VARIABLE to get the value of the variable >>VARIABLE<<
print-%  : ; @echo $* = $($*) ## get the value of a makefile variable '%' (type make print-VARIABLE to get value of VARIABLE)

.DEFAULT_GOAL=all

-include $(DEP)
