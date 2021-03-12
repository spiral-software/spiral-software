# -*- Mode: shell-script -*-

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(gap.colors);

# The package provides coloring of output on capable terminals via
# ANSI escape sequences. It doesn't provide a way to determine if the
# terminal supports this.

AnsiColors := rec(
    Red := "[1;31m",
    Green := "[1;32m",
    Yellow := "[1;33m",
    Blue := "[1;34m",
    Magenta := "[1;35m",
    Cyan := "[1;36m",
    White := "[1;37m",

    DarkRed := "[31m",
    DarkGreen := "[32m",
    DarkYellow := "[33m",
    DarkBlue := "[34m",
    DarkMagenta := "[35m",
    DarkCyan := "[36m",
    DarkWhite := "[37m",

    # Background colors
    BgRed := "[1;41m",
    BgGreen := "[1;42m",
    BgYellow := "[1;43m",
    BgBlue := "[1;44m",
    BgMagenta := "[1;45m",
    BgCyan := "[1;46m",
    BgWhite := "[1;47m",

    BgDarkRed := "[41m",
    BgDarkGreen := "[42m",
    BgDarkYellow := "[43m",
    BgDarkBlue := "[44m",
    BgDarkMagenta := "[45m",
    BgDarkCyan := "[46m",
    BgDarkWhite := "[47m",

    # Inverse color functions
    IRed := "[1;7;31m",
    IGreen := "[1;7;32m",
    IYellow := "[1;7;33m",
    IBlue := "[1;7;34m",
    IMagenta := "[1;7;35m",
    ICyan := "[1;7;36m",
    IWhite := "[1;7;37m",

    Neutral :=  "[0m"
);

# Colorer factory highliting function
_AnsiColorerString := c ->
    (arg -> String(ConcatList(Concat(c,arg,[AnsiColors.Neutral]), a->String(a))));

_AnsiColorerPrint := c ->
    (arg -> Print(DoForAll(arg, x->Print(c, x)), AnsiColors.Neutral));

Red     := _AnsiColorerPrint(AnsiColors.Red);
Green   := _AnsiColorerPrint(AnsiColors.Green);
Yellow  := _AnsiColorerPrint(AnsiColors.Yellow);
Blue    := _AnsiColorerPrint(AnsiColors.Blue);
Magenta := _AnsiColorerPrint(AnsiColors.Magenta);
Cyan    := _AnsiColorerPrint(AnsiColors.Cyan);
White   := _AnsiColorerPrint(AnsiColors.White);

DarkRed     := _AnsiColorerPrint(AnsiColors.DarkRed);
DarkGreen   := _AnsiColorerPrint(AnsiColors.DarkGreen);
DarkYellow  := _AnsiColorerPrint(AnsiColors.DarkYellow);
DarkBlue    := _AnsiColorerPrint(AnsiColors.DarkBlue);
DarkMagenta := _AnsiColorerPrint(AnsiColors.DarkMagenta);
DarkCyan    := _AnsiColorerPrint(AnsiColors.DarkCyan);

IRed     := _AnsiColorerPrint(AnsiColors.IRed);
IGreen   := _AnsiColorerPrint(AnsiColors.IGreen);
IYellow  := _AnsiColorerPrint(AnsiColors.IYellow);
IBlue    := _AnsiColorerPrint(AnsiColors.IBlue);
IMagenta := _AnsiColorerPrint(AnsiColors.IMagenta);
ICyan    := _AnsiColorerPrint(AnsiColors.ICyan);
IWhite   := _AnsiColorerPrint(AnsiColors.IWhite);

# --

RedStr     := _AnsiColorerString(AnsiColors.Red);
GreenStr   := _AnsiColorerString(AnsiColors.Green);
YellowStr  := _AnsiColorerString(AnsiColors.Yellow);
BlueStr    := _AnsiColorerString(AnsiColors.Blue);
MagentaStr := _AnsiColorerString(AnsiColors.Magenta);
CyanStr    := _AnsiColorerString(AnsiColors.Cyan);
WhiteStr   := _AnsiColorerString(AnsiColors.White);

DarkRedStr     := _AnsiColorerString(AnsiColors.DarkRed);
DarkGreenStr   := _AnsiColorerString(AnsiColors.DarkGreen);
DarkYellowStr  := _AnsiColorerString(AnsiColors.DarkYellow);
DarkBlueStr    := _AnsiColorerString(AnsiColors.DarkBlue);
DarkMagentaStr := _AnsiColorerString(AnsiColors.DarkMagenta);
DarkCyanStr    := _AnsiColorerString(AnsiColors.DarkCyan);

IRedStr     := _AnsiColorerString(AnsiColors.IRed);
IGreenStr   := _AnsiColorerString(AnsiColors.IGreen);
IYellowStr  := _AnsiColorerString(AnsiColors.IYellow);
IBlueStr    := _AnsiColorerString(AnsiColors.IBlue);
IMagentaStr := _AnsiColorerString(AnsiColors.IMagenta);
ICyanStr    := _AnsiColorerString(AnsiColors.ICyan);
IWhiteStr   := _AnsiColorerString(AnsiColors.IWhite);
