
/* InDebugMode - EvTab hooked and checking for breakpoints if non zero.    */
extern UInt InDebugMode;
/* InBreakpoint - in breakpoint handler if non zero.			   */
extern UInt InBreakpoint;

/****************************************************************************
**
*F  void DbgBreak( message, arg1, arg2 )  . . . . . . . break to dbg prompt 
** 
**  DbgBreak stops execution and enters dbg> loop. This function should be 
**  called instead of Error() if you need just to stop execution and instantly
**  provide debugging functions ( for example as response to Ctrl+C).
*/

extern void DbgBreak(char* message, Int arg1, Int arg2);


extern void CheckBreakpoints(Obj hdE, Obj hdExec);

extern void InitDebug();

/* just a notification that gap enters brk> or dbg> loop */
extern void DbgErrorLoopStarting();

