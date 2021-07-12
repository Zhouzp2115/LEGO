#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../utils/dataloader.hpp"
#include "../../ml/linear.hpp"
#include "../../ml/toplinear.hpp"
#include "../../ml/softmax.hpp"
#include "../../ml/conv2d.hpp"
#include "../../ml/smaxpool2d.hpp"

class Smodel
{
public:
    Smodel(SmcContext *context, int batch_size = 128)
    {
        this->context = context;
        conv1 = new SConv2D(context, 1, 6, 5, batch_size, 28);
        conv1->genTriplets(1);
        pool1 = new SMaxPool2D(context, 2);
        conv2 = new SConv2D(context, 6, 16, 5, batch_size, 12);
        conv2->genTriplets(1);
        pool2 = new SMaxPool2D(context, 2);
        linear1 = new Slinear(context, 500, batch_size, 16 * 4 * 4, 128);
        linear1->init();
        linear2 = new Slinear(context, 500, batch_size, 128, 84);
        linear2->init();
        linear3 = new Slinear(context, 500, batch_size, 84, 10);
        linear3->init();
        sft = new SoftMax(context);
    }

    void genTripletsByClient(int iters)
    {
       conv1->genTriplets(iters);
       conv2->genTriplets(iters);
       linear1->genTripletsByClient(iters);
       linear2->genTripletsByClient(iters);
       linear3->genTripletsByClient(iters);
    }

    Stensor forward(int iter, Stensor sinput)
    {
        int batch_size = sinput.data.sizes()[0];

        sinput = conv1->forward(iter, sinput);
        sinput = pool1->forward(sinput);
        sinput = sinput.truncate(context->ldlen);

        sinput = conv2->forward(iter, sinput);
        sinput = pool2->forward(sinput);
        sinput = sinput.truncate(context->ldlen);

        sinput.data = sinput.data.reshape({batch_size, -1});
        sinput = linear1->forward(iter, sinput);
        sinput = sinput.truncate(context->ldlen);

        sinput = linear2->forward(iter, sinput);
        sinput = sinput.truncate(context->ldlen);
        
        sinput = linear3->forward(iter, sinput);
        Stensor spr = sft->forward_fast(sinput);

        return spr;
    }

    void backward(int iter, Stensor sgrad)
    {
        int batch_size = sgrad.data.sizes()[0];

        sgrad = linear3->backward(iter, sgrad);
        sgrad = sgrad.truncate(context->ldlen);

        sgrad = linear2->backward(iter, sgrad);
        sgrad = sgrad.truncate(context->ldlen);

        sgrad = linear1->backward(iter, sgrad);
        sgrad = sgrad.truncate(context->ldlen);

        sgrad.data = sgrad.data.reshape({batch_size, 16, 4, 4});
        sgrad = pool2->backward(sgrad);
        sgrad = conv2->backward(iter, sgrad);
        sgrad = sgrad.truncate(context->ldlen);

        sgrad = pool1->backward(sgrad);
        sgrad = conv1->backward(iter, sgrad);
    }
    
    void update(long lr = 4)
    {
        conv1->update(lr);
        conv2->update(lr);
        linear1->update(lr);
        linear2->update(lr);
        linear3->update(lr);
    }
    
    SmcContext *context;
    SConv2D *conv1,*conv2;
    SMaxPool2D *pool1, *pool2;
    Slinear *linear1, *linear2, *linear3;
    SoftMax *sft;
};

class PlaintextMoel
{
public:
    PlaintextMoel(SmcContext *context)
    {
        this->context = context;

        conv1 = new nn::Conv2d(nn::Conv2dOptions(1, 6, 5));
        conv2 = new nn::Conv2d(nn::Conv2dOptions(6, 16, 5));
        linear1 = new nn::LinearImpl(16 * 4 * 4, 128);
        linear2 = new nn::LinearImpl(128, 84);
        linear3 = new nn::LinearImpl(84, 10);

        (*conv1)->to(*context->device);
        (*conv1)->weight.set_requires_grad(true);
        (*conv1)->bias = torch::zeros_like((*conv1)->bias);
        (*conv2)->to(*context->device);
        (*conv2)->weight.set_requires_grad(true);
        (*conv2)->bias = torch::zeros_like((*conv2)->bias);
        linear1->to(*context->device);
        linear1->weight.set_requires_grad(true);
        linear1->bias = torch::zeros_like(linear1->bias);
        linear2->to(*context->device);
        linear2->weight.set_requires_grad(true);
        linear2->bias = torch::zeros_like(linear2->bias);
        linear3->to(*context->device);
        linear3->weight.set_requires_grad(true);
        linear3->bias = torch::zeros_like(linear3->bias);
    }
    
    PlaintextMoel(Smodel *smodel)
    {
        context = smodel->context;

        conv1 = new nn::Conv2d(nn::Conv2dOptions(1, 6, 5));
        conv2 = new nn::Conv2d(nn::Conv2dOptions(6, 16, 5));
        linear1 = new nn::LinearImpl(16 * 4 * 4, 128);
        linear2 = new nn::LinearImpl(128, 84);
        linear3 = new nn::LinearImpl(84, 10);

        (*conv1)->to(*context->device);
        (*conv1)->weight = smodel->conv1->getWeigth().revealFloat();
        (*conv1)->weight.set_requires_grad(true);
        (*conv1)->bias = torch::zeros_like((*conv1)->bias);
        (*conv2)->to(*context->device);
        (*conv2)->weight = smodel->conv2->getWeigth().revealFloat();
        (*conv2)->weight.set_requires_grad(true);
        (*conv2)->bias = torch::zeros_like((*conv2)->bias);
        linear1->to(*context->device);
        linear1->weight = smodel->linear1->getWeight().revealFloat();
        linear1->weight.set_requires_grad(true);
        linear1->bias = torch::zeros_like(linear1->bias);
        linear2->to(*context->device);
        linear2->weight = smodel->linear2->getWeight().revealFloat();
        linear2->weight.set_requires_grad(true);
        linear2->bias = torch::zeros_like(linear2->bias);
        linear3->to(*context->device);
        linear3->weight = smodel->linear3->getWeight().revealFloat();
        linear3->weight.set_requires_grad(true);
        linear3->bias = torch::zeros_like(linear3->bias);
    }

    ~PlaintextMoel()
    {
        delete conv1;
        delete conv2;
        delete linear1;
        delete linear2;
        delete linear3;
    }

    Tensor forward(Tensor input, Tensor label)
    {
        int batchsize = input.sizes()[0];
        Tensor x = (*conv1)->forward(input);
        x = max_pool2d(x, {2, 2});
        x = (*conv2)->forward(x);
        x = max_pool2d(x, {2, 2});

        x = x.view({batchsize, -1});
        x = linear1->forward(x);
        x = linear2->forward(x);
        x = linear3->forward(x);

        x = log_softmax(x, 1);
        x = (x * label * -1).sum() / batchsize;

        return x;
    }

    void update(float lr)
    {
        (*conv1)->weight = (*conv1)->weight.detach() - (*conv1)->weight.grad() * lr;
        (*conv2)->weight = (*conv2)->weight.detach() - (*conv2)->weight.grad() * lr;
        linear1->weight = linear1->weight.detach() - linear1->weight.grad() * lr;
        linear2->weight = linear2->weight.detach() - linear2->weight.grad() * lr;
        linear3->weight = linear3->weight.detach() - linear3->weight.grad() * lr;

        (*conv1)->weight.set_requires_grad(false);
        (*conv2)->weight.set_requires_grad(false);
        linear1->weight.set_requires_grad(false);
        linear2->weight.set_requires_grad(false);
        linear3->weight.set_requires_grad(false);
        (*conv1)->weight = (*conv1)->weight.set_requires_grad(true);
        (*conv2)->weight = (*conv2)->weight.set_requires_grad(true);
        linear1->weight = linear1->weight.set_requires_grad(true);
        linear2->weight = linear2->weight.set_requires_grad(true);
        linear3->weight = linear3->weight.set_requires_grad(true);
    }

    Tensor test(Tensor testdata, Tensor testlabel)
    {
        (*conv1)->weight.set_requires_grad(false);
        (*conv2)->weight.set_requires_grad(false);
        linear1->weight.set_requires_grad(false);
        linear2->weight.set_requires_grad(false);
        linear3->weight.set_requires_grad(false);

        int batchnum = testdata.sizes()[0], batchsize = testdata.sizes()[1];

        Tensor acc = torch::zeros({1}).to(*context->device);
        for (int i = 0; i < batchnum; i++)
        {
            Tensor x = (*conv1)->forward(testdata[i]);
            x = max_pool2d(x, {2, 2});
            x = (*conv2)->forward(x);
            x = max_pool2d(x, {2, 2});

            x = x.view({batchsize, -1});
            x = linear1->forward(x);
            x = linear2->forward(x);
            x = linear3->forward(x);

            x = softmax(x, 1);

            Tensor index = x.argmax(1).reshape({-1}).to(kLong);
            index = one_hot(index, 10).to(kFloat);
            acc += (index * testlabel[i]).sum();
        }

        (*conv1)->weight = (*conv1)->weight.set_requires_grad(true);
        (*conv2)->weight = (*conv2)->weight.set_requires_grad(true);
        linear1->weight = linear1->weight.set_requires_grad(true);
        linear2->weight = linear2->weight.set_requires_grad(true);
        linear3->weight = linear3->weight.set_requires_grad(true);

        return acc / (batchnum * batchsize);
    }

private:
    SmcContext *context;
    nn::Conv2d *conv1;
    nn::Conv2d *conv2;
    nn::LinearImpl *linear1;
    nn::LinearImpl *linear2;
    nn::LinearImpl *linear3;
};

int main(int argc, char **argv)
{
    /***** init context *****/
    int role = atoi(argv[1]);
    char *addr = argv[2];
    int port = atoi(argv[3]);
    SmcContext *context = new SmcContext(role, addr, port);
    context->setLD(16);

    torch::manual_seed(role);

    /***** load  data *****/
    int batchSize = 128;
    MnistDataLoader loader(context);
    loader.slice(batchSize);
    loader.data = loader.data.reshape({-1, batchSize, 1, 28, 28});
    loader.testData = loader.testData.reshape({-1, 100, 1, 28, 28});
    loader.reveal();

    cout << "data" << endl
         << loader.realTrainData.max() << endl
         << loader.realTrainData.min() << endl;
    cout << "label" << endl
         << loader.realTrainLabel.max() << endl
         << loader.realTrainLabel.min() << endl;

    //init model
    PlaintextMoel *model = new PlaintextMoel(context);

    for (int epoch = 0; epoch < 0; epoch++)
    {
        for (int i = 0; i < loader.iters; i++)
        {
            Tensor loss = model->forward(loader.realTrainData[i], loader.realTrainLabel[i]);
            loss.backward();
            //model->update(0.0625);
            model->update(0.1);

            if (i % 100 == 0)
            {
                Tensor acc = model->test(loader.realTestData, loader.realTestLabel);
                cout << "acc" << endl
                     << acc << endl;
            }
        }
    }
    
    Smodel *smodel = new Smodel(context, batchSize);
    float all_time = 0;
    for (int epoch = 0; epoch < 10; epoch++)
    {
        for (int i = 0; i < loader.iters; i++)
        {       
            auto start = clock_start();
            if (i % 3 == 0)
                smodel->genTripletsByClient(3);

            Stensor spr = smodel->forward(i, Stensor(loader.data[i], context));
            Stensor sgrad = spr - Stensor(loader.label[i], context);
            smodel->backward(i, sgrad);
            smodel->update(4 + 7 + context->ldlen);
            all_time += time_from(start) / 1000000;
    
            if (i % 3 == 0)
            {
                cout << epoch << " " << i << endl;
                cout << "all_time time cost " << all_time << endl;
            
                PlaintextMoel pmodel = PlaintextMoel(smodel);
                Tensor acc = pmodel.test(loader.realTestData, loader.realTestLabel);
                cout << "acc" << endl
                     << acc << endl;
            }
        }
    }

    delete model;
    delete context;
    return 0;
}
