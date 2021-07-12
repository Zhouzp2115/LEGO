#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../ml/ml.h"

int main(int argc, char **argv)
{
    int *data = new int[60000 * 28 * 28];
    int *label = new int[60000];
    int *test = new int[10000 * 28 * 28];
    int *test_label = new int[10000];
    unsigned char num;
    FILE *file;
    
    //read train data
    file = fopen("/root/ppml-cpp/bin/FASHIONMNIST/train-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 60000 * 28 * 28; i++)
    {
        num = fgetc(file);
        data[i] = (unsigned int)num;
    }

    //read train label
    fclose(file);
    file = fopen("/root/ppml-cpp/bin/FASHIONMNIST/train-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 60000; i++)
    {
        num = fgetc(file);
        label[i] = (unsigned int)num;
    }

    //read test data
    fclose(file);
    file = fopen("/root/ppml-cpp/bin/FASHIONMNIST/t10k-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 10000 * 28 * 28; i++)
    {
        num = fgetc(file);
        test[i] = (unsigned int)num;
    }

    //read test label
    fclose(file);
    file = fopen("/root/ppml-cpp/bin/FASHIONMNIST/t10k-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 10000; i++)
    {
        num = fgetc(file);
        test_label[i] = (unsigned int)num;
    }


    //proc
    Tensor trainTensor = torch::from_blob(data, {60000, 1, 28, 28}, kInt32).to(kFloat32);
    Tensor labelTensor = torch::from_blob(label, {60000}, kInt32).to(kLong);
    labelTensor = torch::one_hot(labelTensor, 10).to(kLong);
    Tensor testTensor = torch::from_blob(test, {10000, 1, 28, 28}, kInt32).to(kFloat32);
    Tensor testLabelTensor = torch::from_blob(test_label, {10000}, kInt32).to(kLong);
    testLabelTensor = torch::one_hot(testLabelTensor, 10).to(kLong);

    Tensor min = trainTensor.min();
    Tensor max = trainTensor.max();
    Tensor mid = (max - min) / 2.0;
    trainTensor = (trainTensor - mid) / mid;

    min = testTensor.min();
    max = testTensor.max();
    mid = (max - min) / 2.0;
    testTensor = (testTensor - mid) / mid;

    //secert share
    //init context
    int role = atoi(argv[1]);
    char *addr = argv[2];
    int port = atoi(argv[3]);
    SmcContext *context = new SmcContext(role, addr, port);
    context->setLD(20);
    long ld = context->ld;

    if (role == ALICE)
    {
        trainTensor = (trainTensor * ld).to(*context->device).to(kLong);
        labelTensor = (labelTensor * ld).to(*context->device).to(kLong);
        testTensor = (testTensor * ld).to(*context->device).to(kLong);
        testLabelTensor = (testLabelTensor * ld).to(*context->device).to(kLong);

        //train data
        Tensor random = torch::randint(-1 * ld, ld, trainTensor.sizes()).to(kLong).to(*context->device);
        trainTensor = trainTensor - random;
        context->netServer->sendTensor(random);
        torch::save(trainTensor.cpu(), "./FASHIONMNIST/train_data_alice.pt");

        //train label
        random = torch::randint(-1 * ld, ld, labelTensor.sizes()).to(kLong).to(*context->device);
        labelTensor = labelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(labelTensor.cpu(), "./FASHIONMNIST/train_label_alice.pt");

        //test data
        random = torch::randint(-1 * ld, ld, testTensor.sizes()).to(kLong).to(*context->device);
        testTensor = testTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testTensor.cpu(), "./FASHIONMNIST/test_data_alice.pt");

        //testLabel
        random = torch::randint(-1 * ld, ld, testLabelTensor.sizes()).to(kLong).to(*context->device);
        testLabelTensor = testLabelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testLabelTensor.cpu(), "./FASHIONMNIST/test_label_alice.pt");
    }
    else
    {
        Tensor recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./FASHIONMNIST/train_data_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./FASHIONMNIST/train_label_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./FASHIONMNIST/test_data_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./FASHIONMNIST/test_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] test;
    delete[] test_label;
}