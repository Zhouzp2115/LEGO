import time
from sympy import *
from sympy import S
from sympy.core.numbers import mod_inverse
from phe import paillier


def fastExpMod(b, e, m):
    result = 1
    while e != 0:
        if (e & 1) == 1:
            # ei = 1, then mul
            result = (result * b) % m
        e = e // 2
        # b, b^2, b^4, b^8, ... , b^(2^n)
        b = (b*b) % m
    return result


def printhex(x):
    str = '['
    while x > 0:
        byte = x & 0xff
        x = x >> 8
        hex_byte = hex(byte)
        hex_byte = hex_byte[2:len(hex_byte)]
        if len(hex_byte) < 2:
            str = str + '0x0' + hex_byte + ','
        else:
            str = str + '0x' + hex_byte + ','
            
    str = str[0:len(str) - 1] + ']'
    print(str)

publickey_file = open('./publickey.byte' ,'wb+')
privatekey_file = open('./privatekey.byte' ,'wb+')

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
#public_key.n = 40349
#public_key.g = 40350
print('public key n')
print(public_key.n)
print(hex(public_key.n))
printhex(public_key.n)
n_bytes = public_key.n.to_bytes(256 ,'little')
publickey_file.write(n_bytes)
privatekey_file.write(n_bytes)


print('\npublic key n_2')
print(public_key.nsquare)
print(hex(public_key.nsquare))
printhex(public_key.nsquare)
nsquare_byte = public_key.nsquare.to_bytes(256 ,'little')
privatekey_file.write(nsquare_byte)
publickey_file.write(nsquare_byte)
publickey_file.close()

print('\npublic key g')
print(public_key.g)
print(hex(public_key.g))
printhex(public_key.g)

print('\nprivate key p')
print(private_key.p)
print(hex(private_key.p))
printhex(private_key.p)
nsquare_byte = private_key.p.to_bytes(256 ,'little')
privatekey_file.write(nsquare_byte)

print('\nprivate key q')
print(private_key.q)
print(hex(private_key.q))
printhex(private_key.q)
nsquare_byte = private_key.q.to_bytes(256 ,'little')
privatekey_file.write(nsquare_byte)

print('\nprivate key lamda')
lamda = lcm(private_key.p - 1, private_key.q - 1)
lamda = int(lamda)
print(lamda)
print(hex(lamda))
printhex(lamda)
nsquare_byte = lamda.to_bytes(256 ,'little')
privatekey_file.write(nsquare_byte)

print('\nprivate key g_lamda_inv')
g_lamda = (fastExpMod(public_key.g, lamda, public_key.nsquare) - 1) // public_key.n
g_lamda_inv = mod_inverse(g_lamda, public_key.n)
print(g_lamda_inv)
print(hex(g_lamda_inv))
printhex(g_lamda_inv)
nsquare_byte = g_lamda_inv.to_bytes(256 ,'little')
privatekey_file.write(nsquare_byte)

#print(public_key.encrypt(value=3, r_value=3).ciphertext(False))
