/****************************************************************************
**
*A  comments.h                  SPIRAL source                Yevgen Voronenko
**
*A  $Id: comments.h 9926 2010-11-19 06:04:39Z vlad $
**
**  This package implements special handling of comments.
*/

void InitCommentBuffer();
void ClearCommentBuffer();
void AppendCommentBuffer(char *text, int tlen);
Obj  GetCommentBuffer();
Obj  FunClearCommentBuffer(Obj hdCall);
Obj  FunCommentBuffer(Obj hdCall);
Obj  FunDoc(Obj hdCall);
Obj  FunDocLoc(Obj hdCall);
Obj  FunAllDocs(Obj hdCall);

Obj  FunSaveDefLine(Obj hdCall);

void DocumentValue( Obj val, Obj comment );
void DocumentVariable( Obj var, Obj comment );

extern Obj MakeDefString();
extern Obj FindDocString(Obj hd);
extern Int ExtractDefinedIn( char* text, char* fileName, Int* line);
/*
*F FindDocAndExtractLoc . . . extracting fileName and line number 
** from documentation associated with obj.
** Returns 1 if found; 0 if no documentation found; -1 if location
** information is not found in documentation.
*/
extern Int FindDocAndExtractLoc(Bag obj, char* fileName, Int* line);

/* 1 = append [[defined ...]] to docstrings
   0 = do not append 
*/
extern UInt SAVE_DEF_LINE;

/* __doc__ recname */
extern Obj HdDocRecname;
