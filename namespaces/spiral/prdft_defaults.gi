
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



# These options should be used for compiling PRDFT
#
PRDFTDefaults := CopyFields(SpiralDefaults, rec(
	formulaStrategies  := rec(
               sigmaSpl    := [ sigma.StandardSumsRules, sigma.HfuncSumsRules ], 
	       rc          := [ sigma.StandardSumsRules, sigma.HfuncSumsRules ], 
	       preRC       := [],
               postProcess := []
	),

	compileStrategy := compiler.SimpleCS,
));

# BBDefaults := unr -> CopyFields(SpiralDefaults, rec(
# 	formulaStrategies  := rec(
#                sigmaSpl    := libgen.LibStrategy,
# 	       rc          := sigma.RCStrategy,
# 	       preRC       := [],
#                postProcess := [ 
# 		   (opts, s) -> BlockSums(unr, s), 
# 		   sigma.BBLocalize 
# 	       ]
# 	),

# 	# NOTE: memo variables were breaking with other compile strategies
# 	compileStrategy := compiler.NoCSE,
# 	generateInitFunc := true,

# 	codegen := DefaultCodegen,
# 	unparser := CUnparserProg
# ));
