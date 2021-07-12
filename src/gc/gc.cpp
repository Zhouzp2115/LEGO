#include <sys/time.h>
#include <cstdlib>
#include <cmath>
#include "gc.h"

GC::GC(HighSpeedNetIO  *io, int role, long ldlen)
{
	this->io = io;
	this->ldlen = ldlen;
	this->ld = 1l << ldlen;
	this->role = role;

	auto ctx = setup_semi_honest(io, role);
	//ctx->set_batch_size(1024 * 1024); //set larger BOB input processing batch size

	sender = new IKNPAsyn<HighSpeedNetIO >(io);
	recver = new IKNPAsyn<HighSpeedNetIO >(io);

	if (role == ALICE)
	{
		sender->asyn_setup_send(100 * 10000);
		recver->asyn_setup_recv(100 * 10000);
	}
	else
	{
		recver->asyn_setup_recv(100 * 10000);
		sender->asyn_setup_send(100 * 10000);
	}
}

GC::~GC()
{
	//io->flush();
	//delete io;
	delete sender;
	delete recver;
}

void GC::set_ld(long ldlen)
{
	this->ldlen = ldlen;
	this->ld = 1l << ldlen;
}

HighSpeedNetIO  *GC::get_io()
{
	return io;
}

void GC::ot_send(uint64_t *a0, uint64_t *a1, long length)
{
	auto start = clock_start();
	block *b0 = new block[length];
	block *b1 = new block[length];

	for (int i = 0; i < length; i++)
	{
		b0[i] = makeBlock(0, a0[i]);
		b1[i] = makeBlock(0, a1[i]);
	}

	sender->send(b0, b1, length);
	io->flush();

	delete[] b0;
	delete[] b1;
	//cout << "ot send time cost " << time_from(start) * 1.0/ 1000000 << endl;
}

void GC::ot_send_relu(long *a0, long *a1, bool *signal, long length)
{
	auto start = clock_start();
	block *b0 = new block[length];
	block *b1 = new block[length];

	for (int i = 0; i < length; i++)
	{
		if (signal[i])
		{
			b0[i] = makeBlock(0, a0[i]);
			b1[i] = makeBlock(0, a1[i]);
		}
		else
		{
			b1[i] = makeBlock(0, a0[i]);
			b0[i] = makeBlock(0, a1[i]);
		}
	}

	sender->send(b0, b1, length);
	io->flush();

	delete[] b0;
	delete[] b1;
	//cout << "ot send time cost " << time_from(start) / 1000000 << endl;
}

uint64_t *GC::ot_recv(bool *choice, long length)
{
	auto start = clock_start();
	block *r = new block[length];

	recver->recv(r, choice, length);
	io->flush();

	uint64_t *res = new uint64_t[length];
	for (int i = 0; i < length; i++)
	{
		uint64_t *v64val = (uint64_t *)&(r[i]);
		res[i] = v64val[0];
	}

	delete[] r;
	//cout << "ot recv time cost " << time_from(start) / 1000000 << endl;
	return res;
}

long *GC::gc_argmax(long *input, long batch, long kind, long bitlen)
{
	int indexBitLen = (int)(log(kind) / log(2) + 1);
	Integer *A0 = new Integer[batch * kind];
	Integer *A1 = new Integer[batch * kind];
	Integer *A = new Integer[batch * kind];
	Integer *maxIndex = new Integer[batch];
	Integer *index = new Integer[batch * kind];

	for (int i = 0; i < batch * kind; i++)
	{
		A0[i] = Integer(bitlen, input[i], ALICE);
		A1[i] = Integer(bitlen, input[i], BOB);
		index[i] = Integer(indexBitLen, i % kind, PUBLIC);
	}

	for (int i = 0; i < batch * kind; i++)
		A[i] = A0[i] + A1[i];

	for (int i = 0; i < batch; i++)
	{
		//find max
		maxIndex[i] = Integer(indexBitLen, 0, PUBLIC);
		Integer max = A[i * kind];
		for (int j = 1; j < kind; j++)
		{
			Integer tmp = A[i * kind + j];
			Bit geq = true;
			Bit to_swap = ((max < tmp) == geq);
			swap(to_swap, max, tmp);
			swap(to_swap, maxIndex[i], index[i * kind + j]);
		}
	}

	long *reveal = gc_reveal(maxIndex, batch, indexBitLen, PUBLIC);
	io->flush();

	delete[] A0;
	delete[] A1;
	delete[] A;
	delete[] maxIndex;
	delete[] index;

	return reveal;
}

bool *GC::gc_compare(long *input, long length, long bitlen)
{
	Integer *A0 = new Integer[length];
	Integer *A1 = new Integer[length];
	Integer *A = new Integer[length];
	bool *signal = new bool[length];

	for (long i = 0; i < length; i++)
	{
		A0[i] = Integer(bitlen, input[i], ALICE);
		A1[i] = Integer(bitlen, input[i], BOB);
	}

	for (long i = 0; i < length; i++)
		A[i] = A0[i] + A1[i];
	
	for (int i = 0; i < length; i++)
		signal[i] = getLSB(A[i].bits[bitlen - 1].bit);
    
	bool *recved_signal = new bool[length];
	if(role == ALICE)
	{
		io->send_data(signal, length);
		io->recv_data(recved_signal ,length);
	}
	else
	{
		io->recv_data(recved_signal, length);
		io->send_data(signal, length);
	}
    io->flush();

	for (long i = 0; i < length; i++)
		signal[i] = !(signal[i] ^ recved_signal[i]);
	
	delete[] A0;
	delete[] A1;
	delete[] A;
	delete[] recved_signal;

	return signal;
}

long *GC::gc_maxpool(long *input, long length, long pool_size)
{
    auto start = clock_start();
	cout << "maxpool length " << length << endl;
	int indexBitLen = (int)(log(pool_size) / log(2) + 1);
	Integer *A0 = new Integer[length];
	Integer *A1 = new Integer[length];
	Integer *A = new Integer[length];
	Integer *maxIndex = new Integer[length / pool_size];
	Integer *index = new Integer[length];

	for (long i = 0; i < length; i++)
	{
		A0[i] = Integer(64, input[i], ALICE);
		A1[i] = Integer(64, input[i], BOB);
		index[i] = Integer(indexBitLen, i % pool_size, PUBLIC);
	}
	for (long i = 0; i < length; i++)
		A[i] = A0[i] + A1[i];
	for (long i = 0; i < length / pool_size; i++)
		maxIndex[i] = Integer(indexBitLen, 0, PUBLIC);

	cout << "data init time cost " << time_from(start) / 1000000 << endl;

	for (long i = 0; i < length; i += pool_size)
	{
		//find max
		Integer max = A[i];
		for (int j = 1; j < pool_size; j++)
		{
			Bit geq = true;
			Bit to_swap = ((max < A[i + j]) == geq);
			swap(to_swap, max, A[i + j]);
			swap(to_swap, maxIndex[i / pool_size], index[i + j]);
		}
	}
	cout << "find max time cost " << time_from(start) / 1000000 << endl;

	long *grad_ptr = new long[length];
	long *index_ptr = new long[length / pool_size];
	memset(grad_ptr, 0x00, length * sizeof(long));
	for (long i = 0; i < length / pool_size; i++)
		index_ptr[i] = maxIndex[i].reveal<int>();

	cout << "reveal time cost " << time_from(start) / 1000000 << endl;

	for (long i = 0; i < length / pool_size; i++)
		grad_ptr[i * pool_size + index_ptr[i]] = 1;
   
	io->flush();
	delete[] A0;
	delete[] A1;
	delete[] A;
	delete[] index;
	delete[] maxIndex;
	delete[] index_ptr;

	return grad_ptr;
}

bool *GC::gc_relu(long *input, long length, long bitlen)
{
	Integer *A0 = new Integer[length];
	Integer *A1 = new Integer[length];
	Integer *A = new Integer[length];
	bool *signal = new bool[length];

	for (int i = 0; i < length; i++)
	{
		A0[i] = Integer(bitlen, input[i], ALICE);
		A1[i] = Integer(bitlen, input[i], BOB);
	}

	for (int i = 0; i < length; i++)
	{
		A[i] = A0[i] + A1[i];
		signal[i] = getLSB(A[i].bits[bitlen - 1].bit);
	}
    
	delete[] A0;
	delete[] A1;
	delete[] A;
	return signal;
}

void GC::gc_sigmod(long *left, long *right, long length, bool *b3, bool *b4, long bitlen)
{
	auto start = clock_start();

	Integer *A0 = new Integer[length];
	Integer *A1 = new Integer[length];
	Integer *A = new Integer[length];
	Bit *b1 = new Bit[length];
	Bit *b2 = new Bit[length];

	for (int i = 0; i < length; i++)
	{
		A0[i] = Integer(bitlen, left[i], ALICE);
		A1[i] = Integer(bitlen, left[i], BOB);
	}
	//cout << "init data time cost " << time_from(start) / 1000000 << endl;

	for (int i = 0; i < length; i++)
		A[i] = A0[i] + A1[i];
	for (int i = 0; i < length; i++)
		b1[i] = A[i].bits[bitlen - 1];
	//cout << "add cruiuts time cost " << time_from(start) / 1000000 << endl;

	for (int i = 0; i < length; i++)
	{
		A0[i] = Integer(bitlen, right[i], ALICE);
		A1[i] = Integer(bitlen, right[i], BOB);
	}
	//cout << "init data time cost " << time_from(start) / 1000000 << endl;

	for (int i = 0; i < length; i++)
		A[i] = A0[i] + A1[i];
	for (int i = 0; i < length; i++)
		b2[i] = A[i].bits[bitlen - 1];
	//cout << "add cruiuts time cost " << time_from(start) / 1000000 << endl;

	for (int i = 0; i < length; i++)
	{
		Bit b3_bit = !b2[i];
		Bit b4_bit = b2[i] & (!b1[i]);
		b3[i] = getLSB(b3_bit.bit);
		b4[i] = getLSB(b4_bit.bit);
	}
	//cout << "\nget lsb time cost " << time_from(start) / 1000000 << endl;

	io->flush();
	delete[] A0;
	delete[] A1;
	delete[] A;
	delete[] b1;
	delete[] b2;

	//cout << "\ndelete time cost " << time_from(start) / 1000000 << endl;
}

long *GC::gc_reveal(Integer *input, long len, long bitlen, long party)
{
	block *toreveal = new block[len * bitlen];
	block *ptr = toreveal;
	for (int i = 0; i < len; i++)
	{
		memcpy(ptr, (block *)input[i].bits.data(), sizeof(block) * bitlen);
		ptr += bitlen;
	}

	bool *resreveal = new bool[len * bitlen];
	ProtocolExecution::prot_exec->reveal(resreveal, party, toreveal, len * bitlen);

	//recover from bool to long
	long *resLong = new long[len];
	memset(resLong, 0x00, len * sizeof(long));

	for (int i = 0; i < len; i++)
	{
		long tmp = 0;
		tmp = tmp | resreveal[i * bitlen + bitlen - 1];
		for (int j = bitlen - 2; j >= 0; j--)
		{
			tmp = tmp << 1;
			tmp = tmp | resreveal[i * bitlen + j];
		}
		resLong[i] = tmp;
	}

	delete[] toreveal;
	delete[] resreveal;

	return resLong;
}

bool *GC::Y2B(Integer *data, long length)
{
	long bitlen = data[0].bits.size();
	bool *b_share = new bool[bitlen * length];
	bool *index = b_share;

	for (long i = 0; i < length; i++)
	{
		for (long j = 0; j < bitlen; j++)
		{
			index[0] = getLSB(data[i].bits[j].bit);
			index++;
		}
	}

	return b_share;
}

uint64_t *GC::B2A(bool *data, long length, long bitlen)
{
	uint64_t *m1 = new uint64_t[length];
	uint64_t *m0 = new uint64_t[length];

	uint64_t *res = new uint64_t[length / bitlen];
	memset(res, 0x00, length / bitlen * sizeof(int64_t));
	uint64_t *res_index = res;

	for (long i = 0; i < length;)
	{
		for (long j = 0; j < bitlen; j++)
		{
			uint64_t bit = (uint64_t)data[i + j];
			uint64_t r1 = 1 << j;
			uint64_t r0 = (uint64_t)rand();
			r0 = ((r0 << 32) | (uint64_t)rand());

			m1[i + j] = (1 - bit) * r1 + r0;
			m0[i + j] = bit * r1 + r0;
			res_index[0] += r0;
		}
		i += bitlen;
		res_index[0] = (uint64_t)0xFFFFFFFFFFFFFFFF - res_index[0] + 1;
		res_index++;
	}

	if (role == ALICE)
	{
		ot_send(m0, m1, length);
	}
	else
	{
		res_index = res;
		uint64_t *ot_res = ot_recv(data, length);
		for (long i = 0; i < length;)
		{
			res_index[0] = ot_res[i];
			for (long j = 1; j < bitlen; j++)
			{
				res_index[0] += ot_res[i + j];
			}
			res_index++;
			i += bitlen;
		}
		delete[] ot_res;
	}

	delete[] m1;
	delete[] m0;
	return res;
}

uint64_t *GC::Y2A(Integer *data, long length)
{
	bool *b_share = Y2B(data, length);

	long bitlen = data[0].size();
	length = bitlen * length;
	uint64_t *a_share = B2A(b_share, length, bitlen);

	delete[] b_share;
	return a_share;
}

uint64_t *GC::gc_softmax_div(long *numerator, long *denominator, long length, long ldxlen, long zplen)
{
	auto start = clock_start();

	Integer *snumerator = new Integer[length];
	Integer *sdenominator = new Integer[length];
	Integer *numerator1 = new Integer[length];
	Integer *numerator2 = new Integer[length];
	Integer *denominator1 = new Integer[length];
	Integer *denominator2 = new Integer[length];

	for (int i = 0; i < length; i++)
	{
		numerator1[i] = Integer(zplen, numerator[i], ALICE);
		numerator2[i] = Integer(zplen, numerator[i], BOB);
		denominator1[i] = Integer(zplen, denominator[i], ALICE);
		denominator2[i] = Integer(zplen, denominator[i], BOB);
	}

	for (int i = 0; i < length; i++)
	{
		snumerator[i] = numerator1[i] + numerator2[i];
		sdenominator[i] = (denominator1[i] + denominator2[i]) >> ldxlen;
	}
	//cout << "add data time cost " << time_from(start) * 1.0 / 1000000 << endl;

	for (int i = 0; i < length; i++)
		snumerator[i] = snumerator[i] / sdenominator[i];
	//cout << "div time cost " << time_from(start) * 1.0 / 1000000 << endl;

	delete[] sdenominator;
	delete[] numerator1;
	delete[] numerator2;
	delete[] denominator1;
	delete[] denominator2;

	uint64_t *res = Y2A(snumerator, length);
	//cout << "Y2A max time cost " << time_from(start) * 1.0 / 1000000 << endl;
	delete[] snumerator;

	io->flush();

	return res;
}
