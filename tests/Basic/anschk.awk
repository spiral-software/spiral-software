BEGIN { print "\n"; }

/Test:/ {
    line = $0;
    ##  printf("Length line = %d, start of pattern = %d\n", length(line), index(line, "Test:"));
    outb = substr(line, index(line, "Test:"), (length(line) - index(line, "Test:")));
    gsub(/ spiral>/, "", outb);
    gsub(/ >/, "", outb);
    printf "%s\n", outb;
    
    next;
}
