project(ix_c C)

set(SOURCES ix_file.c ix_entry.c ix_error.c)
add_library(ix STATIC ${SOURCES})

add_subdirectory(test)
