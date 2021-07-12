#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../utils/dataloader.hpp"
#include "../../ml/linear.hpp"
#include "../../ml/toplinear.hpp"
#include "../../ml/sigmod.hpp"

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
        pr = (out + 0.5).clamp(0, 1);
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
        
        float acc = 0;
        int batchSize = data.sizes()[1];

        for (int i = 0; i < data.sizes()[0]; i++)
        {
            Tensor input = data[i];
            Tensor batchlabel = label[i].to(kLong).cpu();

            Tensor tpr = forward(input);
            tpr = tpr.cpu();
            float *pr_data = tpr.data<float>();
            long *label_data = batchlabel.data<long>();
            for (int i = 0; i < batchSize; i++)
            {
                if(label_data[i] == 0 && pr_data[i] < 0.5)
                    acc += 1;
                if(label_data[i] == 1 && pr_data[i] > 0.5)
                    acc += 1;
            }
        }

        acc = acc / (data.sizes()[0] * data.sizes()[1]);

        Tensor acc_tensor = torch::zeros({1}, *context->device);
        acc_tensor[0] = acc;
        return acc_tensor;
        
        /*
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
        */
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
    context->setLD(12);

    /***** train on plaintext *****/
    int batchSize = 32;
    
    BinMnistDataLoader loader(context);
    loader.slice(batchSize);
    
    /*
    loader.reveal();
    cout << "data" << endl
         << loader.realTrainData.max() << endl
         << loader.realTrainData.min() << endl;
    cout << "label" << endl
         << loader.realTrainLabel.max() << endl
         << loader.realTrainLabel.min() << endl;
    */

    Tensor acc = torch::zeros({loader.iters, 1}, *context->device);
    PlaintextModel plainModel(context, 784, 1);
    for (int epoch = 0; epoch < 0; epoch++)
    {
        for (int i = 0; i < loader.iters; i++)
        {
            acc[i] = plainModel.test(loader.realTestData, loader.realTestLabel);
            if (i % 10 == 0)
                cout << epoch << " " << i << " acc" << endl
                     << acc[i] << endl;

            plainModel.forward(loader.realTrainData[i]);
            plainModel.backward(loader.realTrainLabel[i]);
            plainModel.update(0.0625 / batchSize);
        }
    }
    acc = acc.cpu();
    float *acc_data = acc.data<float>();
    cout << "------------ acc ---------------" << endl;
    for (int i = 0; i < loader.iters; i++)
        printf("%f,", acc_data[i]);

    cout << "\n-----------------------------" << endl;
    acc = acc.to(*context->device);
    
    //securely train
    TopSlinear linear(context, 60000, batchSize, 784, 1);
    Sigmod sig(context);

    float all_time = 0;
    auto start = clock_start();

    linear.revealMaskedX(loader.data);
    cout << "revealMaskedX time cost " <<  time_from(start) / 1000000 << endl;
    
    start = clock_start();
    int all_iters = 2000000 / batchSize;
    for (int i = 0; i < all_iters; i++)
    {
        /*
        if (i == loader.iters)
            break;

        int index = i % loader.iters;

        plainModel.setWeight(linear.getWeight().revealFloat());
        acc[index] = plainModel.test(loader.realTestData, loader.realTestLabel);
        if (i % 1 == 0)
            cout << i << " acc " << endl
                 << acc[index] << endl;
        */
        int index = i % loader.iters;
        Stensor spr = linear.forward(index);
        spr = sig.forward(spr);
        Stensor sgrad = spr - Stensor(loader.label[index], context) * context->ld;
        linear.backward(index, sgrad);
        //lr = 2^-7--128
        linear.update(4 + 7 + context->ldlen * 2);

        if ((i + 1) % 100 == 0)
            cout << "iter " << i << " time cost " << time_from(start) / 1000000 << endl;
    }
    all_time += time_from(start) / 1000000;

    acc = acc.cpu();
    acc_data = acc.data<float>();
    cout << "------------ acc ---------------" << endl;
    for (int i = 0; i < loader.iters; i++)
        printf("%f,", acc_data[i]);

    cout << "\nall time cost " << all_time << endl;
    delete context;

    return 0;
}