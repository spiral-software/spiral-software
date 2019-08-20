/* Kill '[' and ']' from stdin to stdout
   SE, MP, 16.8.96 
   compile with: cc -o leonin leonin.c
*/

#include <stdio.h>

int main(int argc, char *argv[], char *envp[]) {
  int c;

  c = getchar();
  while (c != EOF) {
    if ((c != '[') && (c != ']'))
      putchar(c);
    c = getchar();
  }
  return 0;
}
