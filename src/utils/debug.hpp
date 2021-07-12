#ifndef __DUBUG__H__
#define __DUBUG__H__

#include "../base/smcbase.h"
#include "../base/stensor.h"

void printfTensor(Tensor tensor)
{
    tensor = tensor.cpu();
    long *data = tensor.data<long>();
    for (int i = 0; i < tensor.numel(); i++)
        cout << data[i] << " ";
}

#endif