#include <sys/time.h>
#include <unistd.h>
#include "../base/smcbase.h"
#include "../base/stensor.h"

void process_binmnist_data(SmcContext *context)
{
    int *data = new int[60000 * 28 * 28];
    int *label = new int[60000];
    int *test = new int[10000 * 28 * 28];
    int *test_label = new int[10000];
    unsigned char num;
    FILE *file;

    //read train data
    file = fopen("./BINMNIST/train-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 60000 * 28 * 28; i++)
    {
        num = fgetc(file);
        data[i] = (unsigned int)num;
    }

    //read train label
    fclose(file);
    file = fopen("./BINMNIST/train-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 60000; i++)
    {
        num = fgetc(file);
        label[i] = (unsigned int)num;
        if (label[i] != 0)
            label[i] = 1;
    }

    //read test data
    fclose(file);
    file = fopen("./BINMNIST/t10k-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 10000 * 28 * 28; i++)
    {
        num = fgetc(file);
        test[i] = (unsigned int)num;
    }

    //read test label
    fclose(file);
    file = fopen("./BINMNIST/t10k-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 10000; i++)
    {
        num = fgetc(file);
        test_label[i] = (unsigned int)num;
        if (test_label[i] != 0)
            test_label[i] = 1;
    }

    //proc
    Tensor trainTensor = torch::from_blob(data, {60000, 1, 28, 28}, kInt32).to(kFloat32);
    Tensor labelTensor = torch::from_blob(label, {60000, 1}, kInt32).to(kLong);
    Tensor testTensor = torch::from_blob(test, {10000, 1, 28, 28}, kInt32).to(kFloat32);
    Tensor testLabelTensor = torch::from_blob(test_label, {10000, 1}, kInt32).to(kLong);

    Tensor min = trainTensor.min();
    Tensor max = trainTensor.max();
    //Tensor mid = (max - min) / 2.0;
    //trainTensor = (trainTensor - mid) / mid;
    trainTensor = trainTensor / max;

    min = testTensor.min();
    max = testTensor.max();
    //mid = (max - min) / 2.0;
    //testTensor = (testTensor - mid) / mid;
    testTensor = testTensor / max;

    //secert share
    long ld = context->ld;
    if (context->role == ALICE)
    {
        trainTensor = (trainTensor * ld).to(*context->device).to(kLong);
        labelTensor = (labelTensor * ld).to(*context->device).to(kLong);
        testTensor = (testTensor * ld).to(*context->device).to(kLong);
        testLabelTensor = (testLabelTensor * ld).to(*context->device).to(kLong);

        //train data
        Tensor random = torch::randint(-1 * ld, ld, trainTensor.sizes()).to(kLong).to(*context->device);
        trainTensor = trainTensor - random;
        context->netServer->sendTensor(random);
        torch::save(trainTensor.cpu(), "./BINMNIST/train_data_alice.pt");

        //train label
        random = torch::randint(-1 * ld, ld, labelTensor.sizes()).to(kLong).to(*context->device);
        labelTensor = labelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(labelTensor.cpu(), "./BINMNIST/train_label_alice.pt");

        //test data
        random = torch::randint(-1 * ld, ld, testTensor.sizes()).to(kLong).to(*context->device);
        testTensor = testTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testTensor.cpu(), "./BINMNIST/test_data_alice.pt");

        //testLabel
        random = torch::randint(-1 * ld, ld, testLabelTensor.sizes()).to(kLong).to(*context->device);
        testLabelTensor = testLabelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testLabelTensor.cpu(), "./BINMNIST/test_label_alice.pt");
    }
    else
    {
        Tensor recv = context->netServer->recvTensor(trainTensor);
        torch::save(recv.cpu(), "./BINMNIST/train_data_bob.pt");

        recv = context->netServer->recvTensor(labelTensor);
        torch::save(recv.cpu(), "./BINMNIST/train_label_bob.pt");

        recv = context->netServer->recvTensor(testTensor);
        torch::save(recv.cpu(), "./BINMNIST/test_data_bob.pt");

        recv = context->netServer->recvTensor(testLabelTensor);
        torch::save(recv.cpu(), "./BINMNIST/test_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] test;
    delete[] test_label;
}

void process_gisette_data(SmcContext *context)
{
    int *data = new int[6000 * 5000];
    int *label = new int[6000];
    int *valid = new int[1000 * 5000];
    int *valid_label = new int[1000];
    char num[4];
    FILE *file;

    file = fopen("./GISETTE/gisette_train.data", "r");
    for (long i = 0; i < 6000; i++)
    {
        for (long j = 0; j < 5000; j++)
        {
            memset(num, 0x00, 4);
            int index = 0;
            while (true)
            {
                char ch = fgetc(file);
                if (ch < 0x30 || ch > 0x39)        
                    break;
               
                num[index] = ch;
                index++;
            }
            data[i * 5000 + j] = atoi(num);
            
            //read \n
            if (j == 4999 && !feof(file))
                char ch = fgetc(file);
        }
    }
    //cout << data[0] << " " << data[4999] << endl;
    //cout << data[5999 * 5000] << " " << data[6000 * 5000 - 1] << endl;

    fclose(file);
    file = fopen("./GISETTE/gisette_train.labels", "r");
    if (file == NULL)
    {
        cout << "open file error" << endl;
        exit(-1);
    }
    for (long i = 0; i < 6000; i++)
    {
        memset(num, 0x00, 4);
        int index = 0;
        while (true)
        {
            char ch = fgetc(file);
            if (ch == 0x0a)
                break;
            
            num[index] = ch;
            index++;
        }
        label[i] = atoi(num);
    }

    fclose(file);
    file = fopen("./GISETTE/gisette_valid.data", "r");
    if (file == NULL)
    {
        cout << "open file error" << endl;
        exit(-1);
    }
    for (long i = 0; i < 1000; i++)
    {
        for (long j = 0; j < 5000; j++)
        {
            memset(num, 0x00, 4);
            int index = 0;
            while (true)
            {
                char ch = fgetc(file);
                if (ch < 0x30 || ch > 0x39)        
                    break;
               
                num[index] = ch;
                index++;
            }
            valid[i * 5000 + j] = atoi(num);
            
            //read \n
            if (j == 4999 && !feof(file))
                char ch = fgetc(file);
        }
    }
    //cout << valid[0] << " " << valid[4999] << endl;
    //cout << valid[999 * 5000] << " " << valid[1000 * 5000 - 1] << endl;

    fclose(file);
    file = fopen("./GISETTE/gisette_valid.labels", "r");
    if (file == NULL)
    {
        cout << "open file error" << endl;
        exit(-1);
    }
    for (long i = 0; i < 1000; i++)
    {
        memset(num, 0x00, 4);
        int index = 0;
        while (true)
        {
            char ch = fgetc(file);
            if (ch == 0x0a)
                break;
            
            num[index] = ch;
            index++;
        }
        valid_label[i] = atoi(num);
    }
    fclose(file);

    //proc
    Tensor trainTensor = torch::from_blob(data, {6000, 5000}, kInt32).to(kFloat32);
    Tensor labelTensor = torch::from_blob(label, {6000}, kInt32).clamp(-1, 0).to(kLong) + 1;
    Tensor validTensor = torch::from_blob(valid, {1000, 5000}, kInt32).to(kFloat32);
    Tensor validLabelTensor = torch::from_blob(valid_label, {1000}, kInt32).clamp(-1, 0).to(kLong) + 1;

    Tensor min = trainTensor.min();
    Tensor max = trainTensor.max();
    //Tensor mid = (max - min);
    //trainTensor = (trainTensor - mid) / mid;
    trainTensor = trainTensor / max;

    min = validTensor.min();
    max = validTensor.max();
    //mid = (max - min);
    //validTensor = (validTensor - mid) / mid;
    validTensor = validTensor / max;

    //secert share
    long ld = context->ld;
    if (context->role == ALICE)
    {
        trainTensor = (trainTensor * ld).to(*context->device).to(kLong);
        labelTensor = (labelTensor * ld).to(*context->device).to(kLong);
        validTensor = (validTensor * ld).to(*context->device).to(kLong);
        validLabelTensor = (validLabelTensor * ld).to(*context->device).to(kLong);

        //train data
        Tensor random = torch::randint(-1 * ld, ld, trainTensor.sizes()).to(kLong).to(*context->device);
        trainTensor = trainTensor - random;
        context->netServer->sendTensor(random);
        torch::save(trainTensor.cpu(), "./GISETTE/train_data_alice.pt");

        //train label
        random = torch::randint(-1 * ld, ld, labelTensor.sizes()).to(kLong).to(*context->device);
        labelTensor = labelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(labelTensor.cpu(), "./GISETTE/train_label_alice.pt");

        //test data
        random = torch::randint(-1 * ld, ld, validTensor.sizes()).to(kLong).to(*context->device);
        validTensor = validTensor - random;
        context->netServer->sendTensor(random);
        torch::save(validTensor.cpu(), "./GISETTE/valid_data_alice.pt");

        //testLabel
        random = torch::randint(-1 * ld, ld, validLabelTensor.sizes()).to(kLong).to(*context->device);
        validLabelTensor = validLabelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(validLabelTensor.cpu(), "./GISETTE/valid_label_alice.pt");
    }
    else
    {
        Tensor recv = context->netServer->recvTensor(trainTensor);
        torch::save(recv.cpu(), "./GISETTE/train_data_bob.pt");

        recv = context->netServer->recvTensor(labelTensor);
        torch::save(recv.cpu(), "./GISETTE/train_label_bob.pt");

        recv = context->netServer->recvTensor(validTensor);
        torch::save(recv.cpu(), "./GISETTE/valid_data_bob.pt");

        recv = context->netServer->recvTensor(validLabelTensor);
        torch::save(recv.cpu(), "./GISETTE/valid_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] valid;
    delete[] valid_label;
}

void process_mnist_data(SmcContext *context)
{
    int *data = new int[60000 * 28 * 28];
    int *label = new int[60000];
    int *test = new int[10000 * 28 * 28];
    int *test_label = new int[10000];
    unsigned char num;
    FILE *file;

    //read train data
    file = fopen("./MNIST/train-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 60000 * 28 * 28; i++)
    {
        num = fgetc(file);
        data[i] = (unsigned int)num;
    }

    //read train label
    fclose(file);
    file = fopen("./MNIST/train-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 60000; i++)
    {
        num = fgetc(file);
        label[i] = (unsigned int)num;
    }

    //read test data
    fclose(file);
    file = fopen("./MNIST/t10k-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 10000 * 28 * 28; i++)
    {
        num = fgetc(file);
        test[i] = (unsigned int)num;
    }

    //read test label
    fclose(file);
    file = fopen("./MNIST/t10k-labels-idx1-ubyte", "r");
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
    //Tensor mid = (max - min) / 2.0;
    //trainTensor = (trainTensor - mid) / mid;
    trainTensor = trainTensor / max;

    min = testTensor.min();
    max = testTensor.max();
    //mid = (max - min) / 2.0;
    //testTensor = (testTensor - mid) / mid;
    testTensor = testTensor / max;

    //secert share
    long ld = context->ld;

    if (context->role == ALICE)
    {
        trainTensor = (trainTensor * ld).to(*context->device).to(kLong);
        labelTensor = (labelTensor * ld).to(*context->device).to(kLong);
        testTensor = (testTensor * ld).to(*context->device).to(kLong);
        testLabelTensor = (testLabelTensor * ld).to(*context->device).to(kLong);

        //train data
        Tensor random = torch::randint(-1 * ld, ld, trainTensor.sizes()).to(kLong).to(*context->device);
        trainTensor = trainTensor - random;
        context->netServer->sendTensor(random);
        torch::save(trainTensor.cpu(), "./MNIST/train_data_alice.pt");

        //train label
        random = torch::randint(-1 * ld, ld, labelTensor.sizes()).to(kLong).to(*context->device);
        labelTensor = labelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(labelTensor.cpu(), "./MNIST/train_label_alice.pt");

        //test data
        random = torch::randint(-1 * ld, ld, testTensor.sizes()).to(kLong).to(*context->device);
        testTensor = testTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testTensor.cpu(), "./MNIST/test_data_alice.pt");

        //testLabel
        random = torch::randint(-1 * ld, ld, testLabelTensor.sizes()).to(kLong).to(*context->device);
        testLabelTensor = testLabelTensor - random;
        context->netServer->sendTensor(random);
        torch::save(testLabelTensor.cpu(), "./MNIST/test_label_alice.pt");
    }
    else
    {
        Tensor recv = context->netServer->recvTensor(trainTensor);
        torch::save(recv.cpu(), "./MNIST/train_data_bob.pt");

        recv = context->netServer->recvTensor(labelTensor);
        torch::save(recv.cpu(), "./MNIST/train_label_bob.pt");

        recv = context->netServer->recvTensor(testTensor);
        torch::save(recv.cpu(), "./MNIST/test_data_bob.pt");

        recv = context->netServer->recvTensor(testLabelTensor);
        torch::save(recv.cpu(), "./MNIST/test_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] test;
    delete[] test_label;
}

void process_cifar_data(SmcContext *context)
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

    Tensor max = trainTensor.max();
    trainTensor = trainTensor / max;

    max = testTensor.max();
    testTensor = testTensor / max;

    //secert share
    //init context
    long ld = context->ld;

    if (context->role == ALICE)
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
        Tensor recv = context->netServer->recvTensor(trainTensor);
        torch::save(recv.cpu(), "./CIFAR10/train_data_bob.pt");

        recv = context->netServer->recvTensor(labelTensor);
        torch::save(recv.cpu(), "./CIFAR10/train_label_bob.pt");

        recv = context->netServer->recvTensor(testTensor);
        torch::save(recv.cpu(), "./CIFAR10/test_data_bob.pt");

        recv = context->netServer->recvTensor(testLabelTensor);
        torch::save(recv.cpu(), "./CIFAR10/test_label_bob.pt");
    }

    //delete
    delete[] data;
    delete[] label;
    delete[] test;
    delete[] test_label;
}

int main(int argc, char **argv)
{
    int role = atoi(argv[1]);
    char *addr = argv[2];
    int port = atoi(argv[3]);
    SmcContext *context = new SmcContext(role, addr, port);
    
    //context->setLD(12);
    //process_binmnist_data(context);
    //context->setLD(16);
    //process_gisette_data(context);
    context->setLD(16);
    process_mnist_data(context);
    //context->setLD(16);
    //process_cifar_data(context);

    delete context;
}