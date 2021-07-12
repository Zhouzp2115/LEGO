#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../utils/dataloader.hpp"
#include "../../ml/linear.hpp"
#include "../../ml/toplinear.hpp"
#include "../../ml/softmax.hpp"

class PlaintextModel
{
public:
     PlaintextModel(SmcContext *context, int inputSize, int outputSize)
     {
          this->context = context;
          this->inputSize = inputSize;
          this->outputSize = outputSize;
          plinear1 = new PlaintextLinear(context, inputSize, outputSize);
     }

     void setWeight(Tensor weight)
     {
          plinear1->setWeight(weight);
     }

     Tensor forward(Tensor input)
     {
          Tensor out = plinear1->forward(input);
          /*
          cout << "out" << endl
               << out.max() << endl
               << out.min() << endl;
          */
          pr = softmax(out, 1);
          return pr;
     }

     void backward(Tensor label)
     {
          Tensor grad = pr - label;
          grad = plinear1->backward(grad);
     }

     void update(float lr)
     {
          plinear1->update(lr);
     }

     Tensor testLoss(Tensor data, Tensor label)
     {
          Tensor out = plinear1->forward(data);
          Tensor logpr = log_softmax(out, 1) * -1;
          Tensor loss = (logpr * label).sum() / data.sizes()[0];

          return loss;
     }

     Tensor test(Tensor data, Tensor label)
     {
          Tensor acc = torch::zeros({1}).to(data.device());

          for (int i = 0; i < data.sizes()[0]; i++)
          {
               Tensor input = data[i];
               Tensor batchlabel = label[i];

               Tensor tpr = forward(input);
               tpr = torch::one_hot(tpr.argmax(1).reshape({-1}), 10).to(kFloat);
               acc += (tpr * batchlabel).sum();
          }

          acc = acc / (data.sizes()[0] * data.sizes()[1]);
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
     context->setLD(12);

     /***** train on plaintext *****/
     int batchSize = 128;

     MnistDataLoader loader(context);
     loader.slice(batchSize);
     loader.reveal();

     cout << "data" << endl
          << loader.realTrainData.max() << endl
          << loader.realTrainData.min() << endl;
     cout << "label" << endl
          << loader.realTrainLabel.max() << endl
          << loader.realTrainLabel.min() << endl;
     
     Tensor acc = torch::zeros({loader.iters, 1}, *context->device);
     PlaintextModel plainModel(context, 784, 10);
     //epoch=33 acc=0.92
     for (int epoch = 0; epoch < 1; epoch++)
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
     TopSlinear linear(context, 60000, batchSize, 784, 10);
     SoftMax sft(context);

     float all_time = 0;
     linear.revealMaskedX(loader.data);

     for (int epoch = 0; epoch < 1; epoch++)
     {
          for (int i = 0; i < loader.iters; i++)
          {
               if (i % 1 == 0)
               {
                    plainModel.setWeight(linear.getWeight().revealFloat());
                    acc[i] = plainModel.test(loader.realTestData, loader.realTestLabel);
                    cout << epoch << " " << i << endl
                         << "acc " << acc[i] << endl;
                    //cout << "all time  " << all_time << endl;
               }

               auto start = clock_start();
               Stensor spr = linear.forward(i);
               spr = sft.forward_fast(spr);

               Stensor sgrad = spr - Stensor(loader.label[i], context);
               linear.backward(i, sgrad);
               //lr = 2^-7--128
               linear.update(4 + 7 + context->ldlen);

               all_time += time_from(start) / 1000000;
          }
     }
     plainModel.setWeight(linear.getWeight().revealFloat());
     Tensor acc_ = plainModel.test(loader.realTestData, loader.realTestLabel);
     cout << "acc" << endl
          << acc_ << endl;
     
     acc = acc.cpu();
     acc_data = acc.data<float>();
     cout << "------------ acc ---------------" << endl;
     for (int i = 0; i < loader.iters; i++)
          printf("%f,", acc_data[i]);

     cout << "\nall time cost " << all_time << endl;
     delete context;

     return 0;
}