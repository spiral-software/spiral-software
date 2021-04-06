##
## SPIRAL License
##
## Copyright (c) 2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  Define a function to add a custom command to run SPIRAL to create a source
##  file (i.e., typically run SPIRAL with input spiral/gap code to create a
##  C/C++ source file).  The assumption is that the spiral/gap code creates an
##  output file (e.g., by calling PrintTo) which is then consumed in a later
##  step of the overall build process.  The output file is typically an
##  intermediate target/dependency created during the build process.
##
##  create_source_file (input, target)
##      input  -> must be a qualified native path to the script (e.g., call
##                TO_NATIVE_PATH first).
##      target -> name of target/source file

##  SPIRAL/GAP shell script name
if (WIN32)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral.bat SPIRAL_SCRIPT )
else ()
    include(FindUnixCommands)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral SPIRAL_SCRIPT )
endif ()

function (create_source_file input target)
    if (WIN32)
	add_custom_command (OUTPUT ${target}
			    COMMAND ${SPIRAL_SCRIPT} < ${input}
			    VERBATIM
			    COMMENT "Generating code for ${target}"   )
    else ()
	include (FindUnixCommands)
	if (BASH)
	    add_custom_command (OUTPUT ${target}
				COMMAND ${BASH} -c "${SPIRAL_SCRIPT} < ${input}"
				VERBATIM
				COMMENT "Generating code for ${target}"   )
	else ()
	    message (FATAL_ERROR "Unknown shell: don't know how to build ${target}" )
	endif ()
    endif ()

    add_custom_target ( NAME.${PROJECT_NAME}.${target} ALL
     	DEPENDS ${input}
	VERBATIM )

endfunction ()

