

FIND_LIBRARY(backend REQUIRED)

AUX_SOURCE_DIRECTORY(src TEAM_C_SRC)

ADD_EXECUTABLE(main1 src/main.cpp)
TARGET_LINK_LIBRARIES(main1 backend)

ADD_EXECUTABLE(main2 src/main2.cpp)
TARGET_LINK_LIBRARIES(main2 backend)
