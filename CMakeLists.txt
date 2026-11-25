cmake_minimum_required(VERSION 3.14)

project(DataAnalysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed")

find_package(
    ROOT 
    COMPONENTS 
    Core
    Hist
    HistPainter
    RIO 
    Tree
    Physics
    REQUIRED
)

find_package(
  Eigen3 REQUIRED
)

find_package(
  eudaq_preprocessor
  COMPONENTS
  detector_event
  REQUIRED
)

add_executable(
  dataAnalysis 
  dataAnalyzer.cpp
)

add_executable(
  simAnalysis 
  simAnalyzer.cpp
)

target_include_directories(
  dataAnalysis
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
 )

target_include_directories(
  simAnalysis
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
 )

target_sources(
  dataAnalysis
  PUBLIC
  src/RootTrackAnalyzer.cpp
  src/Painters.cpp
)

target_sources(
  simAnalysis
  PUBLIC
  src/SimRootTrackAnalyzer.cpp
  src/Painters.cpp
)

target_link_libraries(
    dataAnalysis
    PUBLIC
    Eigen3::Eigen
    ROOT::Core
    ROOT::Hist
    ROOT::HistPainter
    ROOT::RIO
    ROOT::Tree
    ROOT::Physics
    /home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so
    eudaq_preprocessor::detector_event
)

target_link_libraries(
    simAnalysis
    PUBLIC
    Eigen3::Eigen
    ROOT::Core
    ROOT::Hist
    ROOT::HistPainter
    ROOT::RIO
    ROOT::Tree
    ROOT::Physics
    /home/romanurmanov/work/TrackingPipeline/ActsTrackingPipeline_build/lib/libSimEventDict.so
    eudaq_preprocessor::detector_event
)
