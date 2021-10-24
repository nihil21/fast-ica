cmake_minimum_required(VERSION 3.20)
project(fast_ica C)

set(CMAKE_C_STANDARD 99)

add_executable(fast_ica src/main.c include/fp.h
        src/utils.c include/utils.h
        src/random.c include/random.h
        src/sorting.c include/sorting.h
        src/tuple.c include/tuple.h
        src/signal.c include/signal.h
        src/matrix.c include/matrix.h
        src/linalg.c include/linalg.h
        src/preprocessing.c include/preprocessing.h
        src/fast_ica.c include/fast_ica.h)
target_link_libraries(fast_ica m)
