/****************************************************************************
**
*A  hooks.h                    SPIRAL source                Yevgen Voronenko
**
*A  $Id: hooks.h 7468 2008-10-29 12:59:11Z vlad $
**
**  Implements user-definable hooks, that are called on special GAP events,
**  for example, when new file is opened for reading.
*/

void HookBeforeOpenInput();
void HookBeforeCloseInput();
void HookAfterOpenInput();
void HookAfterCloseInput();

void HookSessionStart();
void HookSessionEnd();

void HooksBrkHighlightStart();
void HooksBrkHighlightEnd();

void HooksEditFile(char* fileName, int lineNumber);

Obj  HookAssign(Obj assignment);
void InitHooks();
