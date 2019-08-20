/* Convert output of 'desauto -matrix' into GAP-readable 
   SE, 11.3.98 
   compile with: cc -o leonout leonout.c
*/

#include <stdio.h>

 
#define VAR "leontmp"
  /* name of the global GAP variable to be assigned
     when the filtered output file is READ into GAP
  */


int eq(int *x, char *y) { /* are strings the equal? */
  while ((*x != (char)0) && (*y != (char)0)) {
    if (*(x++) != *(y++))
      return 0;
  }
  return 1;
}

void putchars(char *x) { /* puts without '\n' */
  while (*x != (char)0) {
    putchar(*x);
    ++x;
  }
}

int main(int argc, char *argv[], char *envp[]) {
  int c[4], refill;

  /* rub out up to 2nd ";" */
  do {
    c[0] = getchar();
    if (c[0] == EOF)
      return 1;
  } while (c[0] != ';');
  do {
    c[0] = getchar();
    if (c[0] == EOF)
      return 1;
  } while (c[0] != ';');

  /* convert 
       ":  " into " := "     (used for field assignment)
       "g g" into "g_g"      (used in 'strong generators')
       "seq" into VAR ".seq" (predefined to make List)
       "FIN" into "# FIN"    (comment it out)
     copy anything else and add a trailing newline
     to help the GAP reader swallow the input file.
  */
  c[0] = ' ';
  c[1] = ' ';
  c[2] = getchar();
  c[3] = (char)0;
  while (c[2] != EOF) {
    
    refill = 1;
    if (eq(c, ":  ")) { putchars(" := ");     } else
    if (eq(c, "g g")) { putchars("g_g");      } else
    if (eq(c, "seq")) { putchars(VAR ".seq"); } else
    if (eq(c, "FIN")) { putchars("# FIN");    } else
      refill = 0;

    if (refill) {
      c[0] = getchar();
      if (c[0] == EOF) {
        putchar('\n');
        return 0;
      }
      c[1] = getchar();
      if (c[0] == EOF) {
        putchar(c[0]);
        putchar('\n');
        return 0;
      }
      c[2] = getchar();
    } else {
      putchar(c[0]);
      c[0] = c[1];
      c[1] = c[2];
      c[2] = getchar();
    }
  }
  putchar(c[0]);
  putchar(c[1]);
  putchar('\n');

  return 0;
}
