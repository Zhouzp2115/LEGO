#ifndef __GC_H__
#define __GC_H__

#include <emp-ot/emp-ot.h>
#include <emp-tool/emp-tool.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include <iostream>

using namespace emp;
using namespace std;

class GC
{
public:
    GC(HighSpeedNetIO  *io, int role, long ldlen);

    ~GC();

    HighSpeedNetIO  *get_io();

    void set_ld(long ldlen);

    void ot_send(uint64_t *a0, uint64_t *a1, long length);

    void ot_send_relu(long *a0, long *a1, bool *signal, long length);

    uint64_t *ot_recv(bool *choice, long length);

    long *gc_argmax(long *input, long batch, long kind, long bitlen);

    bool *gc_compare(long *input, long length, long bitlen);

    long *gc_maxpool(long *input, long length, long pool_size);

    //gc_relu
    bool *gc_relu(long *input, long length, long bitlen);

    //gc_sigmod
    void gc_sigmod(long *left, long *right, long length, bool *b3, bool *b4, long bitlen);

    uint64_t *Y2A(Integer *data, long length);

    uint64_t *gc_softmax_div(long *numerator, long *denominator, long length, long ldxlen,long zplen);

private:
    bool *Y2B(Integer *data, long length);

    uint64_t *B2A(bool *data, long length, long bitlen);

    long *gc_reveal(Integer *input, long len, long bitlen = 64, long party = ALICE);

    char *address;
    int port, role;
    long ld, ldlen;
    HighSpeedNetIO  *io;
    IKNPAsyn<HighSpeedNetIO > *sender;
    IKNPAsyn<HighSpeedNetIO > *recver;
};
#endif
