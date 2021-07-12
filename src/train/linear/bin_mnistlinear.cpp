#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../utils/dataloader.hpp"
#include "../../ml/linear.hpp"
#include "../../ml/toplinear.hpp"

class PlaintextModel
{
public:
    PlaintextModel(SmcContext *context, int inputSize, int outputSize)
    {
        this->context = context;
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        plinear1 = new PlaintextLinear(context, inputSize, 1);
    }

    void setWeight(Tensor weight)
    {
        plinear1->setWeight(weight);
    }

    Tensor forward(Tensor input)
    {
        Tensor out = plinear1->forward(input);
        pr = out;
        return pr;
    }

    void backward(Tensor label)
    {
        int batchSize = label.sizes()[0];
        Tensor grad = pr - label;
        grad = plinear1->backward(grad);
    }

    void update(float lr)
    {
        plinear1->update(lr);
    }

    Tensor test(Tensor data, Tensor label)
    {
        Tensor acc = torch::zeros({1}).to(data.device()).to(kLong);
        int batchSize = data.sizes()[1];

        for (int i = 0; i < data.sizes()[0]; i++)
        {
            Tensor input = data[i];
            Tensor batchlabel = label[i].to(kLong);

            Tensor tpr = forward(input);
            tpr = (tpr + 0.5).clamp(0, 1).to(kLong);
            acc += (batchSize - (tpr - batchlabel).abs().sum());
        }

        acc = acc.to(kFloat) / (data.sizes()[0] * data.sizes()[1]);
        return acc;
    }

private:
    SmcContext *context;
    int inputSize, outputSize;
    Tensor pr;
    PlaintextLinear *plinear1;
};

int main(int argc, char **argv)
{
    NoGradGuard guard;

    /***** init context *****/
    SmcContext *context = new SmcContext(atoi(argv[1]), argv[2], atoi(argv[3]));
    context->setLD(16);

    /***** train on plaintext *****/
    int batchSize = 128;
    
    BinMnistDataLoader loader(context);
    loader.slice(batchSize);
    
    loader.reveal();
    cout << "data" << endl
         << loader.realTrainData.max() << endl
         << loader.realTrainData.min() << endl;
    cout << "label" << endl
         << loader.realTrainLabel.max() << endl
         << loader.realTrainLabel.min() << endl;

    PlaintextModel plainModel(context, 784, 1);
    for (int epoch = 0; epoch < 0; epoch++)
    {
        for (int i = 0; i < loader.iters; i++)
        {
            plainModel.forward(loader.realTrainData[i]);
            plainModel.backward(loader.realTrainLabel[i]);
            plainModel.update(0.0078 / batchSize);

            if (i % 10 == 0)
            {
                Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);
                cout << epoch << " " << i << " acc" << endl
                     << acc << endl;
            }
        }
    }
     
    context->netServer->resetNetTime();
    //securely train
    TopSlinear linear(context, 60000, batchSize, 784, 1);

    float all_time = 0;
    auto start = clock_start();

    linear.revealMaskedX(loader.data);
    cout << "revealMaskedX time cost " << time_from(start) / 1000000 << endl;

    int all_iters = 2000000 / batchSize;
    for (int i = 0; i < all_iters; i++)
    {
        int index = i % loader.iters;

        Stensor spr = linear.forward(index);
        spr = spr.truncate(context->ldlen);
        Stensor sgrad = spr - Stensor(loader.label[index], context);
        linear.backward(index, sgrad);
        //lr = 2^-7--128 batchsize = 128(2^7)
        linear.update(7 + 7 + context->ldlen);
        /*
        if (i % 100 == 0)
        {
            plainModel.setWeight(linear.getWeight().revealFloat());
            Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);
            cout << epoch << " " << i << " acc" << endl
                 << acc << endl;
        }
        */
    }
    all_time = time_from(start) / 1000000;
    cout << "net time cost " << context->netServer->getNetTime() << endl;
    cout << "all time cost " << all_time << endl;

    plainModel.setWeight(linear.getWeight().revealFloat());
    Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);
    cout << "acc" << endl
         << acc << endl;

    delete context;

    return 0;
}