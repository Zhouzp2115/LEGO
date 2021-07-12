#include <cstdio>
#include <cstring>
#include <cassert>
#include <unistd.h>

typedef __uint8_t uint8_t;

void paillierSetGPUDevice(int id);

class PublicKey
{
public:
    PublicKey(char *key_file);

    ~PublicKey();

    void *getkeys_array_n(int nelts);

    void *getkeys_array_n2(int nelts);

    //m-m_bytes_len byte r-m_bytes_len byte
    //return 256 byte encrypt result
    uint8_t *encrypt(uint8_t *m, uint8_t *r, int nelts, int element_byte_len);

private:
    uint8_t *repeat(uint8_t *input, int nelts);

    uint8_t n[256], n2[256];
    void *array_n, *array_n2;
};

class PrivateKey
{
public:
    PrivateKey(char *key_file);

    ~PrivateKey();

    void *getkeys_array_n(int nelts);

    void *getkeys_array_n2(int nelts);

    void *getkeys_array_lamda(int nelts);

    void *getkeys_array_lg_nv(int nelts);

    //return 256 byte decrypt result
    uint8_t *decrypt(uint8_t *ctx, int nelts);

private:
    uint8_t *repeat(uint8_t *input, int nelts);

    uint8_t n[256], n2[256], p[256], q[256], lamda[256], lg_inv[256];
    void *array_n, *array_n2, *array_p, *array_q, *array_lamda, *array_lg_inv;
};

uint8_t *dot_he(uint8_t *ctx, uint8_t *ptx, uint8_t *r, uint8_t *r_mask, int batch_size, int input_size, PublicKey *pk);
