add_library(hostLibs Batch.cpp WorkerCL.cpp io.cpp dbgl.cpp Data.cpp)

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_PROGRESS TRUE
  GIT_SHALLOW TRUE
  GIT_TAG v3.10.5)

FetchContent_MakeAvailable(json)

target_link_libraries(hostLibs PRIVATE nlohmann_json::nlohmann_json)
