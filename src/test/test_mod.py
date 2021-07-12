from sympy.ntheory.modular import isprime
import math


def fastExpMod(b, e, m):
    result = 1
    while e != 0:
        if (e&1) == 1:
            # ei = 1, then mul
            result = (result * b) % m
        e = e // 2
        # b, b^2, b^4, b^8, ... , b^(2^n)
        b = (b*b) % m
    return result


number = 2**62
while(True):
    prime = isprime(number)
    if(prime):
        break
    else:
        number = number - 1


#4611686018427387847
zp = 4611686018427387847
#ld = 2**16
ld = 100
#ld_ = 4288060163954311115
ld_ = fastExpMod(ld, zp - 2, zp)  
print('zp ', zp)
print('ld ', ld)
print('ld_ ', ld_)
print('ld*ld_ ', ld*ld_ % zp)

#u0 = 2**55 + 2**58
#u1 = 2**54 + 2**57
#v0 = 2**57 + 123457894
#v1 = 2**57 + 123457894
#z0 = 2**59 + 1457894
u00 = 897*10
u01 = 348*10
v00 = 1020*10
v01 = 946*10
z00 = 6879
z01 = ((u00+u01)*(v00+v01) - z00) % zp
u10 = 97*10
u11 = 34*10
v10 = 102*10
v11 = 96*10
z10 = 679
z11 = ((u10+u11)*(v10+v11) - z10) % zp

delta = ((z01 + z00 + z11 + z10) % zp - (u00+u01)
         * (v00+v01) - (u10+u11)*(v10+v11)) % zp
print('u*v-z ', delta)

#a0 = 2**59 + 2**58 + 123474
#a1 = 2**59 + 2**58 + 234574
#b0 = 2**59 + 2**58 + 1345746
#b1 = 2**59 + 2**58 + 1345748

# a = 1.1 b = 0.7
a0 = 0.31
a1 = 0.22
b0 = 0.52
b1 = 0.41

x0 = 0.31
x1 = 0.41
y0 = 0.61
y1 = 0.50

print('a*b', (a0+a1)*(b0+b1))
print('x*y', (x0+x1)*(y0+y1))
print('sum()', (a0+a1)*(b0+b1)+(x0+x1)*(y0+y1))

#a0_ = (int)(a0 * ld) + zp // 2
#a1_ = (int)(a1 * ld) + zp // 2
#b0_ = (int)(b0 * ld) + zp // 2
#b1_ = (int)(b1 * ld) + zp // 2
a0_ld = (int)(a0 * ld) % zp
a1_ld = (int)(a1 * ld) % zp
b0_ld = (int)(b0 * ld) % zp
b1_ld = (int)(b1 * ld) % zp
x0_ld = (int)(x0 * ld) % zp
x1_ld = (int)(x1 * ld) % zp
y0_ld = (int)(y0 * ld) % zp
y1_ld = (int)(y1 * ld) % zp

e0 = (a0_ld + a1_ld - u00 - u01) % zp
f0 = (b0_ld + b1_ld - v00 - v01) % zp
#e0 = (a0_ld + a1_ld - u00 - u01) % zp
#f0 = (b0_ld + b1_ld - v10 - v11) % zp

e1 = (x0_ld + x1_ld - u10 - u11) % zp
f1 = (y0_ld + y1_ld - v10 - v11) % zp
#e1 = (x0_ld + x1_ld - u10 - u11) % zp
#f1 = (y0_ld + y1_ld - v00 - v01) % zp

c00 = (f0*a0_ld + e0*b0_ld + z00) % zp
c01 = (-e0*f0 + f0*a1_ld + e0*b1_ld + z01) % zp
c10 = (f1*x0_ld + e1*y0_ld + z10) % zp
c11 = (-e1*f1 + f1*x1_ld + e1*y1_ld + z11) % zp

print('c00+c01 real ', ((c00 + c01) % zp) / ld / ld)
print('c10+c11 real ', ((c10 + c11) % zp) / ld / ld)
print('c00+c01+c10+c11 real ', ((c00 + c01 + c10 + c11) % zp) / ld / ld)
