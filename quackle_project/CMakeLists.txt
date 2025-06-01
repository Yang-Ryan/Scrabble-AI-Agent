cmake_minimum_required (VERSION 3.10)

project (libquackle)

message("-- Configuring libquackle")

include("${CMAKE_CURRENT_SOURCE_DIR}/Settings.cmake")

find_package(Threads REQUIRED)

# macOS dylib settings
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")

set(LIBQUACKLE_SOURCES
    alphabetparameters.cpp
    bag.cpp
    board.cpp
    boardparameters.cpp
    bogowinplayer.cpp
    catchall.cpp
    clock.cpp
    computerplayer.cpp
    computerplayercollection.cpp
    datamanager.cpp
    endgame.cpp
    endgameplayer.cpp
    enumerator.cpp
    evaluator.cpp
    game.cpp
    gameparameters.cpp
    generator.cpp
    lexiconparameters.cpp
    move.cpp
    player.cpp
    playerlist.cpp
    preendgame.cpp
    rack.cpp
    reporter.cpp
    resolvent.cpp
    sim.cpp
    strategyparameters.cpp
)

set(LIBQUACKLE_HEADERS
    alphabetparameters.h
    bag.h
    board.h
    boardparameters.h
    bogowinplayer.h
    catchall.h
    clock.h
    computerplayer.h
    computerplayercollection.h
    datamanager.h
    endgame.h
    endgameplayer.h
    enumerator.h
    evaluator.h
    fixedstring.h
    gaddag.h
    game.h
    gameparameters.h
    generator.h
    lexiconparameters.h
    move.h
    player.h
    playerlist.h
    preendgame.h
    rack.h
    reporter.h
    resolvent.h
    sim.h
    strategyparameters.h
    uv.h
)

# Build libquackle shared library
add_library(libquackle SHARED
    ${LIBQUACKLE_SOURCES} ${LIBQUACKLE_HEADERS}
)

target_link_libraries(libquackle Threads::Threads)

target_include_directories(libquackle PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(libquackle PROPERTIES
    OUTPUT_NAME "libquackle"
)

# Add quackleio subdirectory (correct one!)
add_subdirectory(quackleio)
