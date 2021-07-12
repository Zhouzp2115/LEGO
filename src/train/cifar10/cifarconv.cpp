#include <sys/time.h>
#include <unistd.h>
#include "../../base/smcbase.h"
#include "../../base/stensor.h"
#include "../../ml/ml.h"

class SConvModel
{
public:
     SConvModel(SmcContext *context)
     {
          this->context = context;

          sconv1 = new SConv2D(3, 32, 5, context);
          pool1 = new SMaxPool2D(2, context);
          sconv2 = new SConv2D(32, 64, 5, context);
          pool2 = new SMaxPool2D(2, context);
          slinear1 = new SLinear(64 * 5 * 5, 128, context);
          slinear2 = new SLinear(128, 128, context);
          slinear3 = new SLinear(128, 128, context);
          slinear4 = new SLinear(128, 10, context);

          scrt = new CrossEntropyLoss(context);
     }

     ~SConvModel()
     {
          delete sconv1;
          delete pool1;
          delete sconv2;
          delete pool2;
          delete slinear1;
          delete slinear2;
          delete slinear3;
          delete scrt;
          delete slinear4;
     }

     Stensor forward(Stensor sx, Stensor slabel)
     {
          //conv
          sx = sconv1->forward(sx);
          sx = pool1->forward(sx);
          sx = sconv2->forward(sx);
          sx = pool2->forward(sx);

          //linear
          int batchsize = sx.data.sizes()[0];
          sx.data = sx.data.reshape({batchsize, -1});
          sx = slinear1->forward(sx);
          sx = slinear2->forward(sx);
          sx = slinear3->forward(sx);
          sx = slinear4->forward(sx);

          //loss
          sx = scrt->forward(sx, slabel);

          return sx;
     }

     Stensor backward(Stensor slabel)
     {
          int batchsize = slabel.data.sizes()[0];
          
          Stensor grad = scrt->backward(slabel);
          grad = slinear4->backward(grad);
          grad = slinear3->backward(grad);
          grad = slinear2->backward(grad);
          grad = slinear1->backward(grad);

          grad.data = grad.data.reshape({batchsize, 64, 5, 5});
          grad = pool2->backward(grad);
          grad = sconv2->backward(grad);
          grad = pool1->backward(grad);
          grad = sconv1->backward(grad);

          return grad;
     }

     void update(long lr)
     {
          sconv1->update(lr);
          sconv2->update(lr);
          slinear1->update(lr);
          slinear2->update(lr);
          slinear3->update(lr);
          slinear4->update(lr);
     }

     SmcContext *context;
     SConv2D *sconv1;
     SMaxPool2D *pool1;
     SConv2D *sconv2;
     SMaxPool2D *pool2;

     SLinear *slinear1;
     SLinear *slinear2;
     SLinear *slinear3;
     SLinear *slinear4;

     CrossEntropyLoss *scrt;
};

class PlaintextMoel
{
public:
    PlaintextMoel()
    {
        conv1 = new nn::Conv2d(nn::Conv2dOptions(3, 32, 5));
        conv2 = new nn::Conv2d(nn::Conv2dOptions(32, 64, 5));
        linear1 = new nn::LinearImpl(64 * 5 * 5, 128);
        linear2 = new nn::LinearImpl(128, 128);
        linear3 = new nn::LinearImpl(128, 10);
    }

    PlaintextMoel(SConvModel *smodel)
    {
        context = smodel->context;

        conv1 = new nn::Conv2d(nn::Conv2dOptions(3, 32, 5));
        conv2 = new nn::Conv2d(nn::Conv2dOptions(32, 64, 5));
        linear1 = new nn::LinearImpl(64 * 5 * 5, 128);
        linear2 = new nn::LinearImpl(128, 128);
        linear3 = new nn::LinearImpl(128, 10);

        (*conv1)->to(*context->device);
        (*conv1)->weight = smodel->sconv1->getWeight().revealFloat();
        (*conv1)->weight.set_requires_grad(true);
        (*conv1)->bias = torch::zeros_like((*conv1)->bias);
        (*conv2)->to(*context->device);
        (*conv2)->weight = smodel->sconv2->getWeight().revealFloat();
        (*conv2)->weight.set_requires_grad(true);
        (*conv2)->bias = torch::zeros_like((*conv2)->bias);
        linear1->to(*context->device);
        linear1->weight = smodel->slinear1->getWeight().revealFloat();
        linear1->weight.set_requires_grad(true);
        linear1->bias = torch::zeros_like(linear1->bias);
        linear2->to(*context->device);
        linear2->weight = smodel->slinear2->getWeight().revealFloat();
        linear2->weight.set_requires_grad(true);
        linear2->bias = torch::zeros_like(linear2->bias);
        linear3->to(*context->device);
        linear3->weight = smodel->slinear4->getWeight().revealFloat();
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
          x = max_pool2d(x, { 2, 2 });

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
          (*conv1)->weight.set_requires_grad(false);
          (*conv2)->weight.set_requires_grad(false);
          linear1->weight.set_requires_grad(false);
          linear2->weight.set_requires_grad(false);
          linear3->weight.set_requires_grad(false);

          (*conv1)->weight -= (*conv1)->weight.grad() * lr;
          (*conv2)->weight -= (*conv2)->weight.grad() * lr;
          linear1->weight -= linear1->weight.grad() * lr;
          linear2->weight -= linear2->weight.grad() * lr;
          linear3->weight -= linear3->weight.grad() * lr;
          
   
          (*conv1)->weight.grad().zero_();
          (*conv2)->weight.grad().zero_();
          linear1->weight.grad().zero_();
          linear2->weight.grad().zero_();
          linear3->weight.grad().zero_();

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

     void compareGrad(SConvModel *smodel, int role)
     {    
          //conv1
          Tensor delta = (*conv1)->weight.grad() - smodel->sconv1->getGrad().revealFloat();
          delta = delta.abs().max();
          if (role == ALICE)
               cout << "conv1 grad delta" << endl
                    << delta << endl;

          //conv2
          delta = (*conv2)->weight.grad() - smodel->sconv2->getGrad().revealFloat();
          delta = delta.abs().max();
          if (role == ALICE)
               cout << "conv2 grad delta" << endl
                    << delta << endl;

          //slinear1
          delta = linear1->weight.grad() - smodel->slinear1->getGrad().revealFloat();
          delta = delta.abs().max();
          if (role == ALICE)
               cout << "slinear1 grad delta" << endl
                    << delta << endl;

          //slinear2
          delta = linear2->weight.grad() - smodel->slinear2->getGrad().revealFloat();
          delta = delta.abs().max();
          if (role == ALICE)
               cout << "slinear2 grad delta" << endl
                    << delta << endl;

          //slinear3
          delta = linear3->weight.grad() - smodel->slinear3->getGrad().revealFloat();
          delta = delta.abs().max();
          if (role == ALICE)
               cout << "slinear3 grad delta" << endl
                    << delta << endl;
     }

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
    context->multriplets->setK(5);
    context->multriplets->setItemSize(1000);
    context->multriplets->reset();
    context->multriplets->load();

    /***** init model *****/
    long ld = (long)context->ld;
    long slr = (long)(0.1 * ld);

    /***** load data *****/
    Tensor cifarData, cifarLabel, testData, testLabel;
    if (role == ALICE)
    {
        torch::load(cifarData, "./CIFAR10/train_data_alice.pt");
        torch::load(cifarLabel, "./CIFAR10/train_label_alice.pt");
        torch::load(testData, "./CIFAR10/test_data_alice.pt");
        torch::load(testLabel, "./CIFAR10/test_label_alice.pt");
        cifarData = cifarData.to(*context->device);
        cifarLabel = cifarLabel.to(*context->device);
        testData = testData.to(*context->device);
        testLabel = testLabel.to(*context->device);
    }
    else
    {
        torch::load(cifarData, "./CIFAR10/train_data_bob.pt");
        torch::load(cifarLabel, "./CIFAR10/train_label_bob.pt");
        torch::load(testData, "./CIFAR10/test_data_bob.pt");
        torch::load(testLabel, "./CIFAR10/test_label_bob.pt");
        cifarData = cifarData.to(*context->device);
        cifarLabel = cifarLabel.to(*context->device);
        testData = testData.to(*context->device);
        testLabel = testLabel.to(*context->device);
    }
    
    cout << cifarData.sizes() << endl;
    cout << cifarLabel.sizes() << endl;
    cout << testData.sizes() << endl;
    cout << testLabel.sizes() << endl;

    //slice batch
    int batchSize = 100;
    int batchNum = cifarData.sizes()[0] / batchSize;
    cifarData = cifarData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, 3, 32, 32});
    cifarLabel = cifarLabel.slice(0, 0, batchNum * batchSize ).reshape({batchNum, batchSize, -1});
    batchSize = 100;
    batchNum = testData.sizes()[0] / batchSize;
    testData = testData.slice(0, 0, batchNum * batchSize).reshape({batchNum, batchSize, 3, 32, 32});
    testLabel = testLabel.slice(0, 0, batchNum * batchSize ).reshape({batchNum, batchSize, -1});

    Tensor realTrainData = Stensor(cifarData, context).revealFloat();
    Tensor realTrainLabel = Stensor(cifarLabel, context).revealFloat();
    Tensor realTestData = Stensor(testData, context).revealFloat();
    Tensor realTestLabel = Stensor(testLabel, context).revealFloat();

    //init model
    SConvModel *smodel = new SConvModel(context);
    PlaintextMoel *model = new PlaintextMoel();

    int batchsize = realTrainData.sizes()[1], epochnum = 10;
    float lr = 0.05, lrdelta = (lr - 0.0005) / epochnum;
    
    for (int epoch = 0; epoch < 0; epoch++)
    {
        for (int i = 0; i < realTrainData.sizes()[0]; i++)
        {
            Tensor loss = model->forward(realTrainData[i], realTrainLabel[i]);
            loss.backward();
            model->update(lr);
        }
        Tensor acc = model->test(realTestData, realTestLabel);
        cout << "acc" << endl
             << acc << endl;
     }
    

     struct timeval start;
     struct timeval end;
     float alltime = 0;

     for (int epoch = 0; epoch < epochnum; epoch++)
     {
          for (int i = 0; i < cifarData.sizes()[0]; i++)
          {
               gettimeofday(&start, NULL);

               Tensor batchInput = cifarData[i];
               Tensor batchLabel = cifarLabel[i];
               Stensor sinput(batchInput, context);
               Stensor slabel(batchLabel, context);
           
               Stensor sres = smodel->forward(sinput, slabel);
               smodel->backward(slabel);
               
               gettimeofday(&end, NULL);
               
               //compare
               /*
               model = new PlaintextMoel(smodel);
               Tensor loss = model->forward(sinput.revealFloat(), slabel.revealFloat());
               loss.backward();
               model->compareGrad(smodel, context->role);
               delete model;
               */

               smodel->update((long)(lr * context->ld));

               float timer = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
               alltime += timer;
               cout << "epoch " << epoch << " iter " << i << " timecost " << timer << " alltime " << alltime << endl;
               cout << "----------------------------------------------" << endl;
               cout << endl;

               if(i % 3 == -1)
               {
                    model = new PlaintextMoel(smodel);
                    Tensor acc = model->test(realTrainData ,realTrainLabel);
                    cout << "acc" << endl
                         << acc << endl;
               }
          }
     }

     delete model;
     delete smodel;
     delete context;

    return 0;
}