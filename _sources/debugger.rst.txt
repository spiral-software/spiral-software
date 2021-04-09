
.. _debugger:

Debugger
========

Stack-based debugger
Error("msg");
f := (n) -> When(n = 1, Error("at bottom"), n * f(n - 1)); 
f(10);
Top();
n;
Down();
n;
Down();
n;
Up();
n;
n + 1;



TraceFunc(<func>)
UntraceFunc(<func>)

RecursiveFindBug := function(ind, spl, verify_func)
Example:
  
  verif := s -> ForAll(CodeSums(32,s).free(), v->IsBound(v.value) or v in [X,Y]);
  RecursiveFindBug(0,s,verif);

see Dir(gap.debug); in spiral for functions list.


Moving Through Stack
--------------------

While in the brk or dbg loop you can move move up and down the stack and look at
details.


**Top()**

x

**Down()**

y

**Up()**

z

**Backtrace()**

q



Down([<levels>]) - moves stack pointer down (as printed on the screen).
<levels> is how many levels down you want to move stack pointer. Moves one
level down if called without paramenters.

Up([<levels>]) - same as Down() but moves stack pointer up.

Top() - print function/method on current.
  
Backtrace([<levels>]) - print top <levels> items in stack.  Default is 5.


Breakpoints
-----------

  1) Breakpoints - breakpoints list. 

  2) Debug(<bool>) - turns on/off debug mode. In debug mode Spiral checking
     breakpoints each time something is evaluated. It runs slower in this mode
     but you can leave it by executing Debug(false).
     
	 Functions Breakpoint(), StepOver(), StepInto(), StepOut() and ReturnAndBreak() enable debug mode automatically.

  3) Breakpoint(<func>|<method>|NULL) - new breakpoint. Creating a new breakpoint 
     and placing it into Breakpoints list. Returns breakpoint record.
     If NULL specified breakpoint will fire whenever it's possible.

  4) BreakpointOnRead(<var>|<record>|<record>.<field>|RecName("field")) - 
     creating a new read access breakpoint. 

  5) BreakpointOnWrite(<var>|<record>|<record>.<field>|RecName("field")) -
     creating a new write access breakpoint.
          
  6) RemoveBreakpoint([<index>]) - removes breakpoint from Breakpoints list.
     "Index" is a position of breakpoint to remove. Removes all breakpoints 
     if called without arguments.

  7) DbgBreak(<text>) - same as Error() internal function but hits into dbg
     loop. This function always returns HdVoid.

  8) ReturnAndBreak(<obj>) - function for switching from brk to dbg loop.
     Returns given object as result of Error() and placing breakpoint to next
     statement.


  While in dbg loop user can execute statements step by step:

  1) StepOver() - step over current statement.
  2) StepInto() - step into current statement.
  3) StepOut() - step out from current statement.
  4) return; - continue execution.

  Breakpoint record fields:
  1) statement - function or method on which breakpoint fires.

  2) condition - function evaluated if statement match. Spiral will stop execution
     only if this function returns true. By default dummy function generated, 
     you can replace it by yours with the same parameters list.

  3) uncondHits - integer counter. This counter incremented each time statement
     is matched, even if condition function returns false.

  Shortcuts:

  Down() - Ctrl+Down
  Up() - Ctrl+Up 
  Top() - Ctrl+\
  StepOver() - F10 (On OS X use with command key)
  StepInto() - F11 (On OS X use with command key)
  StepOut() - F8 (On OS X use with command key)


  Mac OS ctrl+up/ctrl+down with default Terminal.app:
  
    You must add specific sequences for ctrl-up and ctrl-down, they are not
    there by default. Go to preferences, keyboard, and add sequences
    
    ctrl+up: ESC1;5B
    ctrl+down: ESC1;5A

    where ESC is achieved by pressing the escape key. It will substitute in
    \033 automatically.




