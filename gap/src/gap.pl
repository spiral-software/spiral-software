#!/usr/bin/perl
#    hdSubStr = EVAL( PTR(hdCall)[1] );
#    if ( TYPE(hdSubStr)!=T_STRING)
#        return Error(\"usage: Apropos( <sub_string> )\", 0L,0L);   
#    sub_str = PTR(hdSubStr);
use strict;

my $TYPES = {
    "i" => ["int", "HD_TO_INT", "T_INT"],
    "s" => ["char *", "(char*) PTR", "T_STRING"],
    "b" => ["int", "HdTrue == ", "T_BOOL"]
    };

my ($GapArgs, $CArgs, $NumArgs, $StrArgs);
my ($EvalGapArgs, $CheckGapArgs, $SetCArgs);
my ($param, $Param, $param_ctype, $param_cval, $param_gaptype);
my $FuncName;

if(! @ARGV) {
    print stderr "
Usage: gap.pl <function name> <param1> <param2>
       Generates C stub for a GAP internal function.
       <function name> - is the GAP name for a function
       <paramN>        - [t]name, where [t] is type prefix:
                         i : integer
                         s : string
                         b : boolean

Example: gap.pl SubString sstring ioffset icount

";
    exit 1;
}

$FuncName = shift;
$NumArgs = 1;
foreach $param (@ARGV) {
    ($param_ctype, $param_cval, $param_gaptype) = @{ $TYPES->{substr($param,0,1)} };

    $param = substr $param,1;
    $Param = "\u$param";

    $GapArgs = $GapArgs . "    TypHandle hd$Param;\n";
    $EvalGapArgs = $EvalGapArgs . "    hd$Param = EVAL( PTR(hdCall)[$NumArgs] );\n";
    $CArgs = $CArgs . "    $param_ctype $param;\n";
    $StrArgs = $StrArgs . "<$param>, ";
    $SetCArgs = $SetCArgs . "    $param = $param_cval(hd$Param);\n";
    $CheckGapArgs = $CheckGapArgs . "    if( TYPE(hd$Param) != $param_gaptype ) return Error(usage,0L,0L);\n";
    $NumArgs = $NumArgs + 1;
}
if($StrArgs) {
    substr($StrArgs, length($StrArgs)-2) = "";
}

print "
/****************************************************************************
**
*F  $FuncName( $StrArgs ) . . . . . . . . this function does that
**
** This function does that and that and that and that.
*/
TypHandle       Fun$FuncName ( hdCall )
    TypHandle           hdCall;
{
    char * usage = \"usage: $FuncName( $StrArgs )\";
$GapArgs
$CArgs
    /* get and check the argument                                          */
    if ( SIZE(hdCall) != $NumArgs * SIZE_HD )
        return Error(usage, 0L,0L);

$EvalGapArgs
$CheckGapArgs
$SetCArgs
    return HdVoid;
}

/* add this to initialization function for the package
    InstIntFunc( \"$FuncName\",  Fun$FuncName );
*/

"
;
