# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

option(SYGRAPH_DOCS "Generate documentation" ON)

# Add documentation target
if (SYGRAPH_DOCS)
  find_package(Doxygen REQUIRED)
  # Set the output directory for the documentation
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
  
  # Update the Doxyfile with project-specific information
  set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
  
  set(DOXYGEN_INPUT_FILES "${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/docs/index.md")
  set(DOXYGEN_EXTRA_STYLE "${CMAKE_SOURCE_DIR}/docs/tweaks.css")
  set(DOXYGEN_LOGO_PATH "${CMAKE_SOURCE_DIR}/docs/logo.png")

  file(COPY ${DOXYGEN_LOGO_PATH} DESTINATION ${CMAKE_BINARY_DIR}/docs)

  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # Configure a target for generating the documentation
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
  add_dependencies(sygraph doc)

  install(DIRECTORY ${DOXYGEN_OUTPUT_DIRECTORY}
          DESTINATION share/doc/SYgraph)
endif()