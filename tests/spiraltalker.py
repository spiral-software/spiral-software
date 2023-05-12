
import subprocess as sp
import sys
import os
import time
import uuid


if sys.platform == 'win32':
    SPIRAL_EXE = 'spiral.bat'
else:
    SPIRAL_EXE = 'spiral'
    
    
def exitval2str(v):
    if v == 0:
        return 'EXIT OK'
    elif v == 1:
        return 'EXIT ERRORS'
    else:
        return 'EXIT ABNORMAL'


def speval_list(cmds):
    retl = []
    cmd_iter = iter(cmds)
    
    item = next(cmd_iter, None)
    while item != None:
        with SpiralTalker() as st:
            while item != None:
                v = st.speval(item)
                r = st.exitval()
                if r != None:
                    v += exitval2str(r)
                retl.append(v)
                item = next(cmd_iter, None)
    return retl


class SpiralTalker:

    def __init__(self, logfile=None):
        if isinstance(logfile, str):
            self._del_log = False
            self._logfname = logfile
        else:
            self._del_log = True
            self._logfname = str(uuid.uuid4())
        
        self._outlog  = open(self._logfname, 'w')
        self._inlog   = open(self._logfname, 'r')
        
        os.putenv('SPIRAL_PROMPT', ' ')
        
        self._proc = sp.Popen([SPIRAL_EXE], text=True, 
            stdin=sp.PIPE, stdout=self._outlog, stderr=sp.STDOUT)
        time.sleep(1)
        s = self._inlog.read()
        while s == '':
            time.sleep(.25)
            s = self._inlog.read()

        
    def __del__(self):
        try:
            self._proc.stdin.close()
            self._inlog.close()
            self._outlog.close()
            self._proc.terminate()
            if self._del_log:
                while self._proc.poll() == None:
                    time.sleep(.25)
                os.remove(self._logfname)
        except:
            pass

            
    def __enter__(self):
        return self  


    def __exit__(self, type, value, tb):
        self.__del__()
        return True


    def exitval(self):
        return self._proc.poll()


    def speval(self, cmd):
        self._inlog.read()
        self._proc.stdin.write(cmd + ';\n')
        self._proc.stdin.flush()
        time.sleep(.01)
        retstr = self._inlog.read()
        while retstr == '':
            if self.exitval() != None:
                break
            time.sleep(.01)
            retstr = self._inlog.read()
        return retstr.strip()




            
            
            
            
            

