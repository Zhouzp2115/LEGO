#ifndef __DATA_LOADER_H__
#define __DATA_LOADER_H__

#include "../base/smcbase.h"
#include "../base/stensor.h"

class BinMnistDataLoader
{
public:
    BinMnistDataLoader(SmcContext *context)
    {
       this->context = context;

       if (context->role == ALICE)
       {
           torch::load(data, "./BINMNIST/train_data_alice.pt");
           torch::load(label, "./BINMNIST/train_label_alice.pt");
           torch::load(testData, "./BINMNIST/test_data_alice.pt");
           torch::load(testLabel, "./BINMNIST/test_label_alice.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
       else
       {
           torch::load(data, "./BINMNIST/train_data_bob.pt");
           torch::load(label, "./BINMNIST/train_label_bob.pt");
           torch::load(testData, "./BINMNIST/test_data_bob.pt");
           torch::load(testLabel, "./BINMNIST/test_label_bob.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
    }

    void slice(int batchSize)
    {
        //slice data
        iters = data.sizes()[0] / batchSize;
        cout << iters << endl;
        data = data.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        label = label.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        batchSize = 100;
        int batchNum = testData.sizes()[0] / batchSize;
        testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
        testLabel = testLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
    }

    void reveal()
    {
        realTrainData = Stensor(data, context).revealFloat();
        realTrainLabel = Stensor(label, context).revealFloat();
        realTestData = Stensor(testData, context).revealFloat();
        realTestLabel = Stensor(testLabel, context).revealFloat();
    }

    Tensor data, label, testData, testLabel;
    int iters;
    Tensor realTrainData, realTrainLabel, realTestData, realTestLabel;
private:
    SmcContext *context;
};

class MnistDataLoader
{
public:
    MnistDataLoader(SmcContext *context)
    {
       this->context = context;

       if (context->role == ALICE)
       {
           torch::load(data, "./MNIST/train_data_alice.pt");
           torch::load(label, "./MNIST/train_label_alice.pt");
           torch::load(testData, "./MNIST/test_data_alice.pt");
           torch::load(testLabel, "./MNIST/test_label_alice.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
       else
       {
           torch::load(data, "./MNIST/train_data_bob.pt");
           torch::load(label, "./MNIST/train_label_bob.pt");
           torch::load(testData, "./MNIST/test_data_bob.pt");
           torch::load(testLabel, "./MNIST/test_label_bob.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
    }

    void slice(int batchSize)
    {
        //slice data
        iters = data.sizes()[0] / batchSize;
        cout << "mnist dataloader" << endl
             << "batch:" << batchSize << " iters:" << iters << endl;
        data = data.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        label = label.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        batchSize = 100;
        int batchNum = testData.sizes()[0] / batchSize;
        testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
        testLabel = testLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
    }

    void reveal()
    {
        realTrainData = Stensor(data, context).revealFloat();
        realTrainLabel = Stensor(label, context).revealFloat();
        realTestData = Stensor(testData, context).revealFloat();
        realTestLabel = Stensor(testLabel, context).revealFloat();
    }

    Tensor data, label, testData, testLabel;
    int iters;
    Tensor realTrainData, realTrainLabel, realTestData, realTestLabel;
private:
    SmcContext *context;
};

class GisetteDataLoader
{
public:
    GisetteDataLoader(SmcContext *context)
    {
       this->context = context;

       if (context->role == ALICE)
       {
           torch::load(data, "./GISETTE/train_data_alice.pt");
           torch::load(label, "./GISETTE/train_label_alice.pt");
           torch::load(testData, "./GISETTE/valid_data_alice.pt");
           torch::load(testLabel, "./GISETTE/valid_label_alice.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
       else
       {
           torch::load(data, "./GISETTE/train_data_bob.pt");
           torch::load(label, "./GISETTE/train_label_bob.pt");
           torch::load(testData, "./GISETTE/valid_data_bob.pt");
           torch::load(testLabel, "./GISETTE/valid_label_bob.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
    }

    void slice(int batchSize)
    {
        //slice data
        iters = data.sizes()[0] / batchSize;
        cout << iters << endl;
        data = data.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        label = label.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        batchSize = 100;
        int batchNum = testData.sizes()[0] / batchSize;
        testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
        testLabel = testLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
    }

    void reveal()
    {
        realTrainData = Stensor(data, context).revealFloat();
        realTrainLabel = Stensor(label, context).revealFloat();
        realTestData = Stensor(testData, context).revealFloat();
        realTestLabel = Stensor(testLabel, context).revealFloat();
    }

    Tensor data, label, testData, testLabel;
    int iters;
    Tensor realTrainData, realTrainLabel, realTestData, realTestLabel;
private:
    SmcContext *context;
};

class CifarDataLoader
{
public:
    CifarDataLoader(SmcContext *context)
    {
       this->context = context;

       if (context->role == ALICE)
       {
           torch::load(data, "./CIFAR10/train_data_alice.pt");
           torch::load(label, "./CIFAR10/train_label_alice.pt");
           torch::load(testData, "./CIFAR10/test_data_alice.pt");
           torch::load(testLabel, "./CIFAR10/test_label_alice.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
       else
       {
           torch::load(data, "./CIFAR10/train_data_bob.pt");
           torch::load(label, "./CIFAR10/train_label_bob.pt");
           torch::load(testData, "./CIFAR10/test_data_bob.pt");
           torch::load(testLabel, "./CIFAR10/test_label_bob.pt");
           data = data.to(*context->device);
           label = label.to(*context->device);
           testData = testData.to(*context->device);
           testLabel = testLabel.to(*context->device);
       }
    }

    void slice(int batchSize)
    {
        //slice data
        iters = data.sizes()[0] / batchSize;
        cout << "cifar10 dataloader" << endl
             << "batch:" << batchSize << " iters:" << iters << endl;
        data = data.slice(0, 0, iters * batchSize).reshape({iters, batchSize, 3, 32, 32});
        label = label.slice(0, 0, iters * batchSize).reshape({iters, batchSize, -1});
        batchSize = 100;
        int batchNum = testData.sizes()[0] / batchSize;
        testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, 3, 32, 32});
        testLabel = testLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
    }

    void reveal()
    {
        realTrainData = Stensor(data, context).revealFloat();
        realTrainLabel = Stensor(label, context).revealFloat();
        realTestData = Stensor(testData, context).revealFloat();
        realTestLabel = Stensor(testLabel, context).revealFloat();
    }

    Tensor data, label, testData, testLabel;
    int iters;
    Tensor realTrainData, realTrainLabel, realTestData, realTestLabel;
private:
    SmcContext *context;
};

#endif