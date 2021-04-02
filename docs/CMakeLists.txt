##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

##  Find all the public files from which to extract documentation
##  In due course when there are gap files (*.g or *.gi) or C code files that contain
##  documentation specify them here so that Doxygen can be run to extract it
##  set ( SPIRAL_PUBLIC_FILES ... )

##  Setup locations for Doxygen -- placeholders for now
set ( DOXYGEN_INPUT_DIR ${SPIRAL_SOURCE_DIR}/gap/src ${SPIRAL_SOURCE_DIR}/namespaces/spiral )
set ( DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen )
set ( DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml )
set ( DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in )
set ( DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile )

##  Set the Doxygen input and output directories in the Doxyfile
configure_file ( ${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY )

##  Doxygen won't create this for us
file ( MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR} ) 

##  Custom command to run Doxygen (activate if/when Doxygen info to be extracted)
##  Regenerate Doxygen when the Doxyfile or public headers change
##  add_custom_command ( OUTPUT ${DOXYGEN_INDEX_FILE}
##  		DEPENDS ${SPIRAL_PUBLIC_FILES}
##          	COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
##  		MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
##          	COMMENT "Generating SPIRAL docs"
##  		VERBATIM )

##  Named target for Doxygen so we can run the job easily (enable when something to process)
##  add_custom_target ( Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE} )

set ( SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} )
set ( SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx )
set ( SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html )

##  Regenerate Sphinx when:
##    - Doxygen has rerun 
##    - Our doc files have been updated
##    - The Sphinx config has been updated

add_custom_command ( OUTPUT ${SPHINX_INDEX_FILE}
	COMMAND 
		${SPHINX_EXECUTABLE} -b html
		##  Tell Breathe where to find the Doxygen output
		-Dbreathe_projects.SPIRAL=${DOXYGEN_OUTPUT_DIR}/xml
		${SPHINX_SOURCE} ${SPHINX_BUILD}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	DEPENDS 
		##  Other docs files you want to track should go here (or in some variable)
		${CMAKE_CURRENT_SOURCE_DIR}/index.rst
		##  ${DOXYGEN_INDEX_FILE}
	MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
	COMMENT "Generating documentation with Sphinx")

##  Named target so we can run the job easily
add_custom_target ( Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE}
                    COMMENT "Documentation is available in ${SPHINX_BUILD}" )

##  include ( GNUInstallDirs )
##  install ( DIRECTORY ${SPHINX_BUILD}
##  	  DESTINATION ${CMAKE_INSTALL_DOCDIR} )

