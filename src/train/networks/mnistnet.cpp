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
          plinear1 = new PlaintextLinear(context, inputSize, 128);
          plinear2 = new PlaintextLinear(context, 128, 128);
          plinear3 = new PlaintextLinear(context, 128, outputSize);

          //init weight to nozeros
          plinear1->init();
          plinear2->init();
          plinear3->init();
     }

     ~PlaintextModel()
     {
          delete plinear1;
          delete plinear2;
          delete plinear3;
     }

     Tensor forward(Tensor input)
     {
          int batch = input.sizes()[0];
          Tensor out = plinear1->forward(input);
          out = relu1.forward(out);
          out = plinear2->forward(out);
          out = relu2.forward(out);
          out = plinear3->forward(out);
          //pr = softmax(out, 1);

          Tensor max = out.max_values(1).reshape({batch, 1}).repeat({1, outputSize});
          max = (out - max).clamp(-4, 0) * 0.25 + 1;
          Tensor sumed = max.sum(1).reshape({batch, 1}).repeat({1, outputSize});
          pr = max / sumed;
          return pr;
     }

     void backward(Tensor label)
     {
          Tensor grad = pr - label;
          grad = plinear3->backward(grad);
          grad = relu2.backward(grad);
          grad = plinear2->backward(grad);
          grad = relu1.backward(grad);
          grad = plinear1->backward(grad);
     }

     void update(float lr)
     {       
          plinear1->update(lr);
          plinear2->update(lr);
          plinear3->update(lr);
          
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
     
     PlaintextLinear *plinear1;
     PlaintextLinear *plinear2;
     PlaintextLinear *plinear3;
     SReluPlaintext relu1;
     SReluPlaintext relu2;
private:
     SmcContext *context;
     int inputSize, outputSize;
     Tensor pr;
};

class SModel
{
public:
     SModel(SmcContext *context, int batchSize)
     {
          this->context = context;
          linear1 = new Slinear(context, 6000, batchSize, 784, 128);
          linear2 = new Slinear(context, 6000, batchSize, 128, 128);
          linear3 = new Slinear(context, 6000, batchSize, 128, 10);
          linear1->init();
          linear2->init();
          linear3->init();
          relu1 = new SRelu(context);
          relu2 = new SRelu(context);
          sft = new SoftMax(context);

          weight1 = linear1->getWeight().data.clone();
          weight2 = linear2->getWeight().data.clone();
          weight3 = linear3->getWeight().data.clone();
     }

     ~SModel()
     {
          delete linear1;
          delete linear2;
          delete linear3;
          delete relu1;
          delete relu2;
          delete sft;
     }

     Stensor forward(int i, Stensor sinput)
     {
          Stensor spr = linear1->forward(i, sinput);
          spr = relu1->forward(spr);
          spr = spr.truncate(context->ldlen);
          spr = linear2->forward(i, spr);
          spr = relu2->forward(spr);
          spr = spr.truncate(context->ldlen);
          spr = linear3->forward(i, spr);
          spr = sft->forward_fast(spr);
          return spr;
     }

     void backward(int i, Stensor sgrad)
     {
          sgrad = linear3->backward(i, sgrad);
          sgrad = relu2->backward(sgrad);
          sgrad = sgrad.truncate(context->ldlen);
          sgrad = linear2->backward(i, sgrad);
          sgrad = relu1->backward(sgrad);
          sgrad = sgrad.truncate(context->ldlen);
          linear1->backward(i, sgrad);
     }
     
     void update(long lr)
     {
          linear1->update(lr);
          linear2->update(lr);
          linear3->update(lr);
     }

     void update_mv(long alpha, long truncate_len)
     {
          linear1->update_mv(alpha, truncate_len);
          linear2->update_mv(alpha, truncate_len);
          linear3->update_mv(alpha, truncate_len);
     }

     Tensor test(PlaintextModel &plainModel, MnistDataLoader &loader)
     {
          plainModel.plinear1->setWeight(linear1->getWeight().revealFloat());
          plainModel.plinear2->setWeight(linear2->getWeight().revealFloat());
          plainModel.plinear3->setWeight(linear3->getWeight().revealFloat());
          Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);
 
          return acc;
     }

     void check(PlaintextModel &plainModel, MnistDataLoader &loader)
     {
          plainModel.plinear1->setWeight(linear1->getWeight().revealFloat());
          plainModel.plinear2->setWeight(linear2->getWeight().revealFloat());
          plainModel.plinear3->setWeight(linear3->getWeight().revealFloat());
          Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);

          acc = acc.cpu();
          float *acc_data = acc.data<float>();
          
          //thresold 0.2
          if(acc_data[0] < 0.2)
          {
               cout << "truncate error (recovery weight)" << endl;
               linear1->setWeight(weight1.clone());
               linear2->setWeight(weight2.clone());
               linear3->setWeight(weight3.clone());

               Tensor acc = test(plainModel ,loader);
               cout << "recovery acc " << endl
                    << acc << endl;
          }
          else
          {
               weight1 = linear1->getWeight().data.clone();
               weight2 = linear2->getWeight().data.clone();
               weight3 = linear3->getWeight().data.clone();
          }
     }

private:
     SmcContext *context;
     Slinear *linear1, *linear2, *linear3;
     SRelu *relu1,*relu2;
     SoftMax *sft;
     
     //for recovery
     Tensor weight1, weight2, weight3;
};

int main(int argc, char **argv)
{
     NoGradGuard guard;

     /***** init context *****/
     SmcContext *context = new SmcContext(atoi(argv[1]), argv[2], atoi(argv[3]));
     context->setLD(16);

     /***** load  data *****/
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
     
     /***** train on plaintext *****/
     PlaintextModel plainModel(context, 784, 10);
     //epoch=15 acc=0.97
     for (int epoch = 0; epoch < 0; epoch++)
     {
          for (int i = 0; i < loader.iters; i++)
          {
               plainModel.forward(loader.realTrainData[i]);
               plainModel.backward(loader.realTrainLabel[i]);
               //plainModel.update(0.0078 / batchSize);
               plainModel.update(0.0625 / batchSize);

               if (i % 100 == 0)
               {
                    Tensor acc = plainModel.test(loader.realTestData, loader.realTestLabel);
                    cout << epoch << " " << i << " acc" << endl
                         << acc << endl;
               }
          }
     }

     //securely train
     SModel smodel(context, batchSize);
     float all_time = 0;

     for (int epoch = 0; epoch < 10; epoch++)
     {
          for (int i = 0; i < loader.iters; i++)
          {
               if (i % 10 == 0)
               {
                    //Tensor acc = smodel.test(plainModel, loader);
                    cout << epoch << " " << i << endl;
                    //     << " acc " << acc << endl;
                    cout << "all time  " << all_time << endl
                         << endl;
               }

               auto start = clock_start();
               Stensor spr = smodel.forward(i, Stensor(loader.data[i], context));
               Stensor sgrad = spr - Stensor(loader.label[i], context);
               smodel.backward(i, sgrad);
               //lr = 2^-7--0.0078  2^-4--0.0625
               smodel.update(4 + 7 + context->ldlen);

               if ((i + 1) % 100 == 0)
                    smodel.check(plainModel, loader);

               all_time += time_from(start) / 1000000;
          }
     }
    
     Tensor acc = smodel.test(plainModel, loader);
     cout << "acc" << endl
          << acc << endl;

     cout << "all time cost " << all_time << endl;
     delete context;

     return 0;
}