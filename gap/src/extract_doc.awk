##  extract documentation from C files to import as Gap strings for documentation in Gap
##  Run as follows:  gawk -f extract_doc.awk gap/src/*.c
##  Don't include the .h files as many things will be multiply declared since much of the
##  GAP legacy code puts the same documentation in two places...

BEGIN   {
    echo_lines = 0;
    fileis = "";
    fun_name = "";
    gap_name = "";
    defined_at = -1;
    FS = "[() ]+";              ## include () as field separator so don't make '(arg...' part of name
}

/^*F/   {
    ##  sub ( /^./, "#", $0 );
    ##  print $0;
    ##  for ( i = 1; i <= NF; i++ ) print "Field " i ": " $i;
    ##  Only extract the doc for function that begin with "Fun" -- these are the implementations for the corresponding Gap keywords
    if ( substr ( $2, 1, 3 ) == "Fun" ) {
        fileis = FILENAME;
        defined_at = FNR;
        fun_name = $2;
        gap_name = substr ( fun_name, 4 );
        echo_lines = 1;
    }
}

/^\*\//         ||              
/[ \*]+\*\//      {
    if ( echo_lines ) {
        printf ( "##\t%s (%s) defined in file: %s, line: %d\n", fun_name, gap_name, fileis, defined_at );
        printf ( "\nDocumentVariable(%s);\nDocumentVariable(%s);\na := 0;\n\n", fun_name, gap_name );
    }
    echo_lines = 0;
}

{
    sub ( /^./, "#", $0 );
    sub ( /$/, "", $0 );
    if ( echo_lines ) print $0;
}
