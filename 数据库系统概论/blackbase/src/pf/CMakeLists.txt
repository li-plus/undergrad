project(pf_c C)

set(SOURCES pf_file.c pf_cache.c pf_hash_table.c pf_error.c)
add_library(pf STATIC ${SOURCES})

add_subdirectory(test)
