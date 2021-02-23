
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Declare(c3, c1);

# MBRDFT cost via Murakami decomposition
c3 := n -> Checked(n >=4, n mod 4 = 0, Cond(
    n=4, [6,2],
    2*c3(n/2) + n/4*c3(4)
));

c1 := n -> Checked(n >= 2, n mod 2 = 0, Cond(
    n=2, [2,0],
    n=4, [6,0],
    c1(n/2) + c3(n/2) + n/2 * c1(2)
));

Declare(r3,r1);
            
# PRDFT cost via rDFT decomposition 
r3 := n -> Checked(n >=4, n mod 4 = 0, Cond(
    n=4, [6,4],
    2*r3(n/2) + n/4*r3(4)
));

r1 := n -> Checked(n >= 2, n mod 2 = 0, Cond(
    n=2, [2,0],
    n=4, [6,0],
    n=8, [20,2],
    r1(2) + ((n-4)/4) * r3(4) + 2 * r1(n/2)
));


            
