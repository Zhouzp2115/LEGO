#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../ml/ml.h"

class SModel
{
public:
     SModel(SmcContext *context)
     {
          this->context = context;

          slinear1 = new SLinear(784, 128, context);
          slinear2 = new SLinear(128, 128, context);
          slinear3 = new SLinear(128, 10, context);
          srelu1 = new SRelu(context);
          srelu2 = new SRelu(context);
          scrt = new CrossEntropyLoss(context);

          time_offline = 0;
          time_online = 0;
          com_size = 0;
     }

     ~SModel()
     {
          delete slinear1;
          delete slinear2;
          delete slinear3;
          delete srelu1;
          delete srelu2;
          delete scrt;
     }

     Stensor forward(Stensor sinput, Stensor slabel)
     {    
          //com size
          context->netServer->reSetData();

          Stensor res = slinear1->forward(sinput);
          res = srelu1->forward(res);
          res = slinear2->forward(res);
          res = srelu2->forward(res);
          res = slinear3->forward(res);
          res = scrt->forward(res, slabel);

          //com size 
          com_size += context->netServer->getDataSize();
          context->netServer->reSetData();

          return res;
     }

     Stensor backward(Stensor slabel)
     {
          //com size
          context->netServer->reSetData();

          Stensor grad = scrt->backward(slabel);
          grad = slinear3->backward(grad);
          grad = srelu2->backward(grad);
          grad = slinear2->backward(grad);
          grad = srelu1->backward(grad);
          grad = slinear1->backward(grad);
          
          //com size 
          com_size += context->netServer->getDataSize();
          context->netServer->reSetData();

          return grad;
     }

     void update(long lr)
     {
          slinear1->update(lr);
          slinear2->update(lr);
          slinear3->update(lr);
     }

     void print()
     {
          if (context->role == ALICE)
          {
               cout << "slinear1 weight max" << endl
                    << slinear1->getWeight().revealFloat().abs().max() << endl;
               cout << "slinear2 weight max" << endl
                    << slinear2->getWeight().revealFloat().abs().max() << endl;
               cout << "slinear3 weight max" << endl
                    << slinear3->getWeight().revealFloat().abs().max() << endl;
          }
          else
          {
               slinear1->getWeight().revealFloat().abs().max();
               slinear2->getWeight().revealFloat().abs().max();
               slinear3->getWeight().revealFloat().abs().max();
          }
     }
     
     float getOfflineTime()
     { 
          time_offline += slinear1->getOfflineTime();
          time_offline += slinear2->getOfflineTime();
          time_offline += slinear3->getOfflineTime();
          time_offline += context->multriplets->getOfflineTime();
          
          slinear1->resetOfflineTime();
          slinear2->resetOfflineTime();
          slinear3->resetOfflineTime();
          context->multriplets->resetOfflineTime();

          return time_offline;
     }

     void resetOfflineTime()
     {
          time_offline = 0;
     }

     SmcContext *context;

     SLinear *slinear1;
     SLinear *slinear2;
     SLinear *slinear3;
     SRelu *srelu1;
     SRelu *srelu2;
     CrossEntropyLoss *scrt;

     float time_offline;
     float time_online;
     float com_size;
};

class PlaintextModel
{
public:
     PlaintextModel(SModel *smodel)
     {
          this->context = smodel->context;
          this->smodel = smodel;

          linear1 = new nn::LinearImpl(784, 128);
          linear2 = new nn::LinearImpl(128, 128);
          linear3 = new nn::LinearImpl(128, 10);
          linear1->to(*context->device);
          linear2->to(*context->device);
          linear3->to(*context->device);
          linear1->weight = smodel->slinear1->getWeight().revealFloat().set_requires_grad(true);
          linear1->bias = torch::zeros_like(linear1->bias);
          linear2->weight = smodel->slinear2->getWeight().revealFloat().set_requires_grad(true);
          linear2->bias = torch::zeros_like(linear2->bias);
          linear3->weight = smodel->slinear3->getWeight().revealFloat().set_requires_grad(true);
          linear3->bias = torch::zeros_like(linear3->bias);
     }

     PlaintextModel(SmcContext *context)
     {
          linear1 = new nn::LinearImpl(784, 128);
          linear2 = new nn::LinearImpl(128, 128);
          linear3 = new nn::LinearImpl(128, 10);
          linear1->to(*context->device);
          linear2->to(*context->device);
          linear3->to(*context->device);
          linear1->bias = torch::zeros_like(linear1->bias);
          linear2->bias = torch::zeros_like(linear2->bias);
          linear3->bias = torch::zeros_like(linear3->bias);
     }

     ~PlaintextModel()
     {
          delete linear1;
          delete linear2;
          delete linear3;
     }

     Tensor forward(Tensor input, Tensor label)
     {
          int batchsize = input.sizes()[0];

          Tensor loss = linear1->forward(input);
          loss = relu(loss);
          loss = linear2->forward(loss);
          loss = relu(loss);
          loss = linear3->forward(loss);
          loss = log_softmax(loss, 1);
          loss = (loss * label * -1).sum() / batchsize;

          return loss;
     }

     void update(float lr)
     {
          linear1->weight.set_requires_grad(false);
          linear2->weight.set_requires_grad(false);
          linear3->weight.set_requires_grad(false);

          linear1->weight -= linear1->weight.grad() * lr;
          linear2->weight -= linear2->weight.grad() * lr;
          linear3->weight -= linear3->weight.grad() * lr;

          linear1->weight.grad().zero_();
          linear2->weight.grad().zero_();
          linear3->weight.grad().zero_();

          linear1->weight.set_requires_grad(true);
          linear2->weight.set_requires_grad(true);
          linear3->weight.set_requires_grad(true);
     }

     Tensor test(Tensor data, Tensor label)
     {
          linear1->weight.set_requires_grad(false);
          linear2->weight.set_requires_grad(false);
          linear3->weight.set_requires_grad(false);

          Tensor acc = torch::zeros({1}).to(data.device());
          int batchSize = data.sizes()[1];
          for (int i = 0; i < data.sizes()[0]; i++)
          {
               Tensor input = data[i].reshape({batchSize, -1});
               Tensor batchlabel = label[i];
               Tensor linearRes = linear1->forward(input);
               linearRes = relu(linearRes);
               linearRes = linear2->forward(linearRes);
               linearRes = relu(linearRes);
               linearRes = linear3->forward(linearRes);
               Tensor pr = softmax(linearRes, 1);
               pr = torch::one_hot(pr.argmax(1).reshape({-1}), 10).to(kFloat);
               acc += (pr * batchlabel).sum();
          }
          acc = acc / (data.sizes()[0] * data.sizes()[1]);

          return acc;
     }

     void compare()
     {
          //compare
          if (context->role == ALICE)
          {
               cout << "linear1 grad delta" << endl
                    << (linear1->weight.grad() - smodel->slinear1->getGrad().revealFloat()).abs().max() << endl;
               cout << "linear2 grad delta" << endl
                    << (linear2->weight.grad() - smodel->slinear2->getGrad().revealFloat()).abs().max() << endl;
               cout << "linear3 grad delta" << endl
                    << (linear3->weight.grad() - smodel->slinear3->getGrad().revealFloat()).abs().max() << endl;
          }
          else
          {
               smodel->slinear1->getGrad().revealFloat();
               smodel->slinear2->getGrad().revealFloat();
               smodel->slinear3->getGrad().revealFloat();
          }
     }

private:
     SmcContext *context;
     SModel *smodel;

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
     context->setLD(20);
     context->multriplets->setK(5);
     context->multriplets->setItemSize(1000);
     context->multriplets->reset();

     /***** load data *****/
     Tensor mnistData, mnistLabel, testData, testLabel;
     if (role == ALICE)
     {
          torch::load(mnistData, "./FASHIONMNIST/train_data_alice.pt");
          torch::load(mnistLabel, "./FASHIONMNIST/train_label_alice.pt");
          torch::load(testData, "./FASHIONMNIST/test_data_alice.pt");
          torch::load(testLabel, "./FASHIONMNIST/test_label_alice.pt");
          mnistData = mnistData.to(*context->device);
          mnistLabel = mnistLabel.to(*context->device);
          testData = testData.to(*context->device);
          testLabel = testLabel.to(*context->device);
     }
     else
     {
          torch::load(mnistData, "./FASHIONMNIST/train_data_bob.pt");
          torch::load(mnistLabel, "./FASHIONMNIST/train_label_bob.pt");
          torch::load(testData, "./FASHIONMNIST/test_data_bob.pt");
          torch::load(testLabel, "./FASHIONMNIST/test_label_bob.pt");
          mnistData = mnistData.to(*context->device);
          mnistLabel = mnistLabel.to(*context->device);
          testData = testData.to(*context->device);
          testLabel = testLabel.to(*context->device);
     }

     //slice batch
     int batchSize = 128;
     int batchNum = mnistData.sizes()[0] / batchSize;
     mnistData = mnistData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, 1, 28, 28});
     mnistLabel = mnistLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});
     batchSize = 100;
     batchNum = testData.sizes()[0] / batchSize;
     testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, 1, 28, 28});
     testLabel = testLabel.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, -1});

     mnistData = mnistData.reshape({468, 128, -1});
     testData = testData.reshape({100, 100, -1});

     Tensor realTrainData = Stensor(mnistData, context).revealFloat();
     Tensor realTrainLabel = Stensor(mnistLabel, context).revealFloat();
     Tensor realTestData = Stensor(testData, context).revealFloat();
     Tensor realTestLabel = Stensor(testLabel, context).revealFloat();

     cout << "traindata max" << endl
          << realTrainData.abs().max() << endl;
     cout << "testdata max" << endl
          << realTestData.abs().max() << endl;

     //init model
     SModel *smodel = new SModel(context);
     PlaintextModel *model = new PlaintextModel(smodel);

     //plaintext
     float plaintextLR = 0.05;
     float delta = (plaintextLR - 0.01) / 10;
     for (int epoch = 0; epoch < 0; epoch++)
     {
          for (int i = 0; i < realTrainData.sizes()[0]; i++)
          {
               Tensor input = realTrainData[i];
               Tensor label = realTrainLabel[i];

               //forward
               Tensor loss = model->forward(input, label);

               loss.backward();

               //update
               model->update(plaintextLR);
          }

          cout << "test acc" << endl;
          Tensor acc = model->test(realTestData, realTestLabel);
          cout << acc << endl;

          //plaintextLR -= delta;
     }
     delete model;

     //mpc
     long lr = (long)(0.05 * context->ld);

     struct timeval start;
     struct timeval end;
     float alltime = 0;

     for (int epoch = 0; epoch < 10; epoch++)
     {
          cout << "epoch " << epoch << endl;

          for (int i = 0; i < mnistData.sizes()[0]; i++)
          {
               gettimeofday(&start, NULL);

               /***** data *****/
               Tensor input = mnistData[i];
               Tensor label = mnistLabel[i];

               /***** mpc *****/
               Stensor sinput(input, context);
               Stensor slabel(label, context);

               //forward
               Stensor sloss = smodel->forward(sinput, slabel);

               //backward
               Stensor sgrad = smodel->backward(slabel);
               gettimeofday(&end, NULL);

               //compare grad
               PlaintextModel *model = new PlaintextModel(smodel);
               Tensor loss = model->forward(realTrainData[i], realTrainLabel[i]);
               loss.backward();

               model->compare();

               //update
               smodel->update(lr); 
               smodel->print();

               cout << "offline time cost " << smodel->getOfflineTime() << endl;
               cout << "com szie " << context->netServer->getDataSize() << endl;

               if (i % 3 == 0)
               {
                    Tensor acc = model->test(realTestData, realTestLabel);
                    cout << "acc" << endl
                         << acc << endl;
               }

               float timer = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
               alltime += timer;
               cout << "epoch " << epoch << " iter " << i << " timecost " << timer << " alltime " << alltime << endl;
               cout << "----------------------------------------------" << endl;
               cout << endl;

               delete model;
          }
     }

     delete smodel;
     delete context;

     return 0;
}