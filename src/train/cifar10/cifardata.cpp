#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../ml/ml.h"

int main(int argc, char **argv)
{
    int *data = new int[3072 * 50000];
    int *label = new int[50000];
    int *test = new int[3072 * 10000];
    int *test_label = new int[10000];
    unsigned char num;
    int count = 0;
    FILE *file;

    //read train data
    int dataIndex = 0, labelIndex = 0;
    for (int i = 1; i < 6; i++)
    {
        stringstream fileName;
        fileName << "./CIFAR10/data_batch_" << i << ".bin";
        cout << fileName.str().c_str() << endl;

        file = fopen(fileName.str().c_str(), "r");

        while (true)
        {
            //read label
            num = fgetc(file);
            label[labelIndex] = (unsigned int)num;
            labelIndex++;

            //read data
            while (true)
            {
                num = fgetc(file);
                data[dataIndex] = (unsigned int)num;
                dataIndex++;
                if (dataIndex % 3072 == 0)
                    break;
            }

            if (labelIndex % 10000 == 0)
                break;
        }

        fclose(file);
    }

    //read test data
    dataIndex = 0;
    labelIndex = 0;
    file = fopen("./CIFAR10/test_batch.bin", "r");
    while (true)
    {
        //read label
        num = fgetc(file);
        test_label[labelIndex] = (unsigned int)num;
        labelIndex++;

        //read data
        while (true)
        {
            num = fgetc(file);
            test[dataIndex] = (unsigned int)num;
            dataIndex++;
            if (dataIndex % 3072 == 0)
                break;
        }

        if (labelIndex % 10000 == 0)
            break;
    }
    fclose(file);


    //proc
    Tensor trainTensor = torch::from_blob(data, {50000, 3, 32, 32}, kInt32).to(kFloat32);
    Tensor labelTensor = torch::from_blob(label, {50000}, kInt32).to(kLong);
    labelTensor = torch::one_hot(labelTensor, 10).to(kLong);
    Tensor testTensor = torch::from_blob(test, {10000, 3, 32, 32}, kInt32).to(kFloat32);
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
        torch::save(trainTensor.cpu(), "./CIFAR10/train_data_alice.pt");

        //train label
        random = torch::randint(-1 * ld, ld, labelTensor.sizes()).to(kLong).to(*context->device);
        labelTensor = labelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(labelTensor.cpu(), "./CIFAR10/train_label_alice.pt");

        //test data
        random = torch::randint(-1 * ld, ld, testTensor.sizes()).to(kLong).to(*context->device);
        testTensor = testTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testTensor.cpu(), "./CIFAR10/test_data_alice.pt");

        //testLabel
        random = torch::randint(-1 * ld, ld, testLabelTensor.sizes()).to(kLong).to(*context->device);
        testLabelTensor = testLabelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testLabelTensor.cpu(), "./CIFAR10/test_label_alice.pt");
    }
    else
    {
        Tensor recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./CIFAR10/train_data_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./CIFAR10/train_label_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./CIFAR10/test_data_bob.pt");

        recv = context->netServer->recvTensor();
        torch::save(recv.cpu(), "./CIFAR10/test_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] test;
    delete[] test_label;
}