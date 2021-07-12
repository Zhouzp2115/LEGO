#include <sys/time.h>
#include <unistd.h>
#include "emp-tool/emp-tool.h"
#include "../base/smcbase.h"
#include "../base/stensor.h"
#include "../ml/linear.hpp"
#include "../ml/toplinear.hpp"
#include "../base/triplets.h"

using namespace torch;
using namespace std;

/****** offline time test *****/
void linear_regression_offline_ot(SmcContext *context)
{
     int n = 20000;
     int d = 500;

     int iters = n / 128;
     auto start = clock_start();
     for (int round = 0; round < 5; round++)
     {
          TripletsOT triplet(context, iters, 128, d, 1);
          triplet.setTopLayer(true);
          triplet.genTriplets();
          cout << "time cost " << time_from(start) * 1.0 / 1000000 << endl;
          cout << "send data " << triplet.getSendData() << endl;
          cout << "recv data " << triplet.getRecvData() << endl;
     }

     cout << n << " " << d << endl;
     cout << "average time cost " << time_from(start) * 1.0 / 1000000 / 5 << endl;
}

void conv_offline_ot(SmcContext *context)
{
     int batch_size = 128;
     int iters = 1;
     TripletsOT conv1(context, iters, batch_size, 784, 128);
     TripletsOT conv2(context, iters, batch_size, 784, 128);
     TripletsOT triplet1(context, iters, batch_size, 784, 512);
     triplet1.setTopLayer(true);
     TripletsOT triplet2(context, iters, batch_size, 512, 512);
     TripletsOT triplet3(context, iters, batch_size, 128, 128);
     TripletsOT triplet4(context, iters, batch_size, 512, 10);
     
     auto start = clock_start();
     //triplet1.genTriplets();
     cout << "triplet1 time cost " << time_from(start) * 1.0 / 1000000 << endl;
     triplet2.genTriplets();
     cout << "triplet2 time cost " << time_from(start) * 1.0 / 1000000 << endl;
     //triplet3.genTriplets();
     cout << "triplet3 time cost " << time_from(start) * 1.0 / 1000000 << endl;
     triplet4.genTriplets();
     cout << "triplet3 time cost " << time_from(start) * 1.0 / 1000000 << endl;
}

void test_conv_offline_ot(SmcContext *context)
{
     int in_channel = 1, out_channel = 32, kernel_size = 5, image_size = 28;
     int batch_size = 128;
     int out_image_size = image_size - kernel_size + 1;
     int n = out_image_size * out_image_size * batch_size;
     int output_size = out_channel;
     int input_size = in_channel * kernel_size * kernel_size;
     TripletsOT triplet1(context, n, batch_size, input_size, output_size);
     
     auto start = clock_start();
     triplet1.genTriplets();
     cout << "time cost " << time_from(start) * 1.0 / 1000000 << endl
                    << endl;

}

void test_LinearTripletsOT(SmcContext *context)
{
     int n[3] = {2000, 20000, 200000};
     int d[3] = {100, 500, 1000};
     int batch[3] = {64, 128, 256};

     for (int i = 0; i < 3; i++)
     {
          for (int j = 0; j < 3; j++)
          {
               TripletsOT triplet(context, n[i], 128, d[j], 1);

               auto start = clock_start();
               triplet.genTriplets();

               cout << n[i] << " " << d[j] << endl;
               cout << "time cost " << time_from(start) * 1.0 / 1000000 << endl
                    << endl;
          }
     }

     
}

/****** base triplets test *****/
void test_TripletsClinet(SmcContext *context)
{
     int batch_size = 128, input_size = 1000, output_size = 2;

     paillierSetGPUDevice(context->role - 1);
     for (int i = 0; i < 1; i++)
     {
          TripletsAdapter *triplet = new TripletsClient(context, 2, batch_size, input_size, output_size);
          auto start = clock_start();
          triplet->genTriplets();
          cout << "time cost " << time_from(start) / 1000000 << endl;
          triplet->valid();
     }
}

void test_TripletsHE(SmcContext *context)
{
     int output_size = 1;
     int batch_size[3] = {64, 128, 256};
     int input_size[3] = {100, 500, 1000};

     paillierSetGPUDevice(context->role - 1);

     for (int i = 0; i < 3; i++)
     {
          for (int j = 0; j < 3; j++)
          {
               TripletsAdapter *triplet = new TripletsHE(context, 1, batch_size[i], input_size[j], output_size);
               triplet->setTopLayer(true);
               auto start = clock_start();
               for (int round = 0; round < 5; round++)
               {
                    triplet->genTriplets();
                    if (round == 0)
                    {
                         cout << "send data " << triplet->getSendData() << endl;
                         cout << "recv data " << triplet->getRecvData() << endl;
                    }
                    //triplet->valid();
               }
               cout << "|B|=" << batch_size[i] << " d=" << input_size[j] << endl;
               cout << "gen time cost " << time_from(start) / 1000000 / 5 << endl;
               cout << endl;
          }
     }
}

void test_TripletsOT(SmcContext *context)
{
     int batch_size = 128, input_size = 1000, output_size = 1;

     for (int i = 0; i < 1; i++)
     {
          TripletsAdapter *triplet = new TripletsOT(context, 15, batch_size, input_size, output_size);
          triplet->setTopLayer(true);
          auto start = clock_start();
          triplet->genTriplets();
          cout << "gen time cost " << time_from(start) / 1000000 << endl;
          cout << "send data " << triplet->getSendData() << endl;
          cout << "recv data " << triplet->getRecvData() << endl;
          triplet->valid();
     }
}

int main(int argc, char **argv)
{
     /***** init context *****/
     int role = atoi(argv[1]);
     char *addr = argv[2];
     int port = atoi(argv[3]);
     SmcContext *context = new SmcContext(role, addr, port);
     context->setLD(16);

     torch::randint(0, 10, {100, 1000}).to(*context->device);
     
     conv_offline_ot(context);

     delete context;
     
     return 0;
}