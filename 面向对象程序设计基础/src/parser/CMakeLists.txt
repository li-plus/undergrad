
FIND_LIBRARY(backend REQUIRED)

INCLUDE_DIRECTORIES(src)
AUX_SOURCE_DIRECTORY(src SIMPLE_PARSER_SRC)

ADD_LIBRARY(parser ${SIMPLE_PARSER_SRC})
TARGET_LINK_LIBRARIES(parser backend)

# add test parser
ADD_EXECUTABLE(test_parser test/testparser.cpp)
TARGET_LINK_LIBRARIES(test_parser parser)
TARGET_LINK_LIBRARIES(test_parser backend)