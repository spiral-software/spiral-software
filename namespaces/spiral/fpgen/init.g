
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(rewrite, spl, code, compiler);

ACM := Concat(Conf("exec_dir"), PATH_SEP, "acm");
Include(acm);
