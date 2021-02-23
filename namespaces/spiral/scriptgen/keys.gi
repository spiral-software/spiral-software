
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#
# keys.gi -- define string constants used as keys in scriptgen package
#


# transforms

SGKEY_FFT		:= "FFT";
SGKEY_FFT_2D	:= "FFT2D";
SGKEY_FFT_3D	:= "FFT3D";
SGKEY_IFFT		:= "IFFT";
SGKEY_IFFT_2D	:= "IFFT2D";
SGKEY_IFFT_3D	:= "IFFT3D";
SGKEY_WHT		:= "WHT";


# data types

SGKEY_DPCX		:= "DPCX";
SGKEY_SPCX		:= "SPCX";


# ScriptGen settings

SGKEY_DATATYPE		:= "datatype";
SGKEY_FILENAME		:= "filename";
SGKEY_FUNCNAME		:= "funcname";
SGKEY_SIZE			:= "size";
SGKEY_TRANSFORM		:= "transform";

# ScriptGen settings details

SGKEY_DISPLAYNAME		:= "displayname";
SGKEY_MULTIPLEVALUES	:= "multipleValues";
SGKEY_NAME				:= "name";
SGKEY_TYPE				:= "type";


# Special keys

SGKEY_DEFAULT	:= "default";


# String constants

SGSTR_STDOUT	:= "*stdout*";

SGSTR_ALL			:= "all";
SGSTR_CONSTRUCTOR	:= "constructor";
SGSTR_RUNRANDOMALL	:= "runRandomAll";
SGSTR_RUNALL		:= "runAll";


# Settings types

SGTYPE_INT		:= "i";
SGTYPE_BOOL		:= "b";
SGTYPE_STRING	:= "s";


# display names associated with keys

_SG_displayNames := rec(
	(SGKEY_FFT)			:= "Fast Fourier Transform",
	(SGKEY_FFT_2D)		:= "2D Fast Fourier Transform",
	(SGKEY_FFT_3D)		:= "3D Fast Fourier Transform",
	(SGKEY_IFFT)		:= "Inverse Fast Fourier Transform",
	(SGKEY_IFFT_2D)		:= "2D Inverse Fast Fourier Transform",
	(SGKEY_IFFT_3D)		:= "2D Inverse Fast Fourier Transform",
	(SGKEY_WHT)			:= "Walsh-Hadamard Transform",
	
	(SGKEY_DPCX)		:= "Double-Precision Complex",
	(SGKEY_SPCX)		:= "Single-Precision Complex",
	
	(SGKEY_DATATYPE)	:= "Data Type",
	(SGKEY_FILENAME)	:= "File Name",
	(SGKEY_FUNCNAME)	:= "Function Name",
	(SGKEY_SIZE)		:= "Size",
	(SGKEY_TRANSFORM)	:= "Transform",
	
	(SGSTR_RUNRANDOMALL) := "Generate Code (quick search)",
	(SGSTR_RUNALL)		 := "Generate Code (in-depth search)",
);


GetScriptGenDisplayName := function(key)
	if IsString(key) and IsBound(_SG_displayNames.(key)) then
		return _SG_displayNames.(key);
	elif (not IsString(key)) and IsList(key) and (Length(key) = 2) then
		if ForAll(key, IsInt) then
			return StringPrint(key[1], " x ", key[2]);
		fi;
	fi;
	
	return StringPrint(key);
end;
	
	
SetScriptGenDisplayName := function(key, string)
	if not ( IsString(key) and IsString(string) ) then
		Error("usage: SetScriptGenDisplayName(key, string)\n  both <key> and <string> must be strings");
	fi;
		
	_SG_displayNames.(key) := string;
end;


# documentation associated with keys

_SG_docunmentationForKeys := rec(
);


GetScriptGenDocumentation := function(key)
	if IsString(key) and IsBound(_SG_docunmentationForKeys.(key)) then
		return _SG_docunmentationForKeys.(key);
	fi;
	
	return "";
end;
	
	
SetScriptGenDocumentation := function(key, string)
	if not ( IsString(key) and IsString(string) ) then
		Error("usage: SetScriptGenDocumentation(key, string)\n  both <key> and <string> must be strings");
	fi;
		
	_SG_docunmentationForKeys.(key) := string;
end;





