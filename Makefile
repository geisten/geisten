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

SOURCE = test_$(PROJECT_NAME).c
HEADER = $(PROJECT_NAME).h
OBJ = $(SOURCE:.c=.o)
DEP = $(OBJ:.o=.d)


# CFLAGS ?= -I. -march=native -mtune=native -MP -Wall -Wextra -mavx -Wstrict-overflow -ffast-math -fsanitize=address -O3 -MMD
CFLAGS ?= -I. -mtune=native -MP -Wall -Wextra -Wstrict-overflow -ffast-math -fsanitize=address -O -MMD -g2

LDFLAGS ?= -ffast-math -fsanitize=address

options:
	@echo $(PROJECT_NAME) build options:
	@echo "CFLAGS   = ${CFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "CC       = ${CC}"

all: options test  ## build all unit tests of project $(PROJECT_NAME)

%.o : %.c
	$(CC) $(CFLAGS) -c $< $(LIB_PATH) $(LIBS) -o $@ $(LDFLAGS)

# build the unit tests
test_%: test_%.o
	$(CC) -o $@ $< $(LDFLAGS)
	./$@ ||  (echo "Test $^ failed" && exit 1)

test: test_$(PROJECT_NAME) ## run all test programs
	@echo "Success, all tests of project '$(PROJECT_NAME)' passed."


.PHONY: clean
# clean the build
clean:  ## cleanup - remove the target (test) files
	rm -f $(OBJ) $(DEP) test_$(PROJECT_NAME) $(PROJECT_NAME).md

.PHONY: install
install: $(PROJECT_NAME).h  ## install the target build to the target directory ('$(DESTDIR)$(PREFIX)/include')
	install $< $(DESTDIR)$(PREFIX)/include/

.PHONY: uninstall
uninstall: ## remove the build from the target directory ('$(DESTDIR)$(PREFIX)/include')
	rm -f $(DESTDIR)$(PREFIX)/include/

# ----------------- TOOLS ------------------------------------------

# c header docu
%.md: %.h
	@$(MKDIR_P) $(dir $@)
	./xtract.awk $< > $@

docs: $(PROJECT_NAME).md ## build the documentation of the header files in markdown format

help: ## print this help information. Type 'make all' to build the project
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
		printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
		}' $(MAKEFILE_LIST)


# just type 'make print-VARIABLE to get the value of the variable >>VARIABLE<<
print-%  : ; @echo $* = $($*) ## get the value of a makefile variable '%' (type make print-VARIABLE to get value of VARIABLE)

.DEFAULT_GOAL=all

-include $(DEP)
