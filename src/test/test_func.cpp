#include <sys/time.h>
#include <unistd.h>
#include <thread>
#include "emp-tool/emp-tool.h"
#include "../base/smcbase.h"
#include "../base/stensor.h"
#include "../ml/srelu.hpp"
#include "../ml/sigmod.hpp"
#include "../ml/toplinear.hpp"
#include "../ml/softmax.hpp"
#include "../ml/linear.hpp"
#include "../ml/conv2d.hpp"
#include "../ml/smaxpool2d.hpp"

using namespace torch;
using namespace std;
using namespace emp;

void test_relu(SmcContext *context)
{
     int batch = 128, features = 128;
     Tensor input = torch::randint(-20, 0, {batch, features}, *context->device).to(kLong) * context->ld;
     Tensor ones = torch::ones({batch, features}, *context->device).to(kLong) * context->ld / 2;

     auto start = clock_start();
     SRelu srelu(context);
     Stensor relued = srelu.forward(Stensor(input, context));
     cout << "relu time cost " << time_from(start) / 1000000 << endl;

     Stensor sgrad = srelu.backward(Stensor(ones, context));
     return;

     cout << "srelu output" << endl
          << relued.revealFloat() << endl;
     cout << "sgrad" << endl
          << relued.revealFloat() << endl;

     input = Stensor(input, context).revealFloat();
     input.set_requires_grad(true);
     Tensor real = relu(input);
     Tensor sumed = real.sum();
     sumed.backward();
     cout << "real relu" << endl
          << real << endl;
     cout << "real grad" << endl
          << input.grad() << endl;
}

void test_sigmod(SmcContext *context)
{
     int batch = 5, kind = 1;
     Tensor input = torch::randint(-1 * context->ld, 1 * context->ld, {kind, batch}, *context->device).to(kLong);
     input = input * context->ld + (1L << 63);

     cout << "input" << endl
          << Stensor(input, context).revealFloat() / context->ld << endl;

     Sigmod sig(context);
     auto start = clock_start();
     Stensor sigmoed = sig.forward(Stensor(input, context));
     cout << "sigmod time cost " << time_from(start) / 1000000 << endl;

     Tensor sres = sigmoed.revealFloat();
     cout << "sres" << endl
          << sres << endl;

     Tensor res = Stensor(input, context).revealFloat() / context->ld;
     res = (res + 0.5).clamp(0, 1);
     cout << "res" << endl
          << res << endl;

     Tensor delta = sres - res;
     cout << "delta" << endl
          << delta.max() << endl
          << delta.min() << endl;
}

void test_softmax(SmcContext *context)
{
     int batch = 2, kind = 10;
     Tensor input = torch::randint(-3, 3, {batch, kind}, *context->device).to(kLong);
     input = input * context->ld * context->ld;

     cout << "input" << endl
          << Stensor(input, context).revealFloat() / context->ld << endl;

     SoftMax sft(context);
     auto start = clock_start();
     Tensor r = torch::randint(0, 1 << 30, {batch, kind}, *context->device).to(kLong);
     r = r.__or__(r.__lshift__(34));
     if (context->role == ALICE)
     {
          input = input - r;
          context->netServer->sendTensor(r);
     }
     else
          input = input + context->netServer->recvTensor(r);
     Stensor spr = sft.forward_fast(Stensor(input, context));
     cout << "spr fast" << endl
          << spr.reveal().to(kFloat) / context->ld << endl;
     spr = sft.forward(Stensor(input, context));
     cout << "spr" << endl
          << spr.reveal().to(kFloat) / context->ld << endl;
     cout << "softmax time cost " << time_from(start) / 1000000 << endl;
     //cout << "spr" << endl
     //     << spr.reveal().to(kFloat) / context->ld << endl;

     //real spr
     input = Stensor(input, context).reveal() / context->ld;
     input = input.to(kFloat) / context->ld;
     Tensor max = input.max_values(1).reshape({batch, 1}).repeat({1, kind});
     input = (input - max).clamp(-4, 0) * 0.25 + 1;
     Tensor sumed = input.sum(1).reshape({batch, 1}).repeat({1, kind});
     Tensor pr = input / sumed;
     cout << "pr" << endl
          << pr;
}

void test_Y2A(SmcContext *context)
{
     long length = 128 * 100;
     Integer *test = new Integer[length];

     for (int i = 0; i < length; i++)
     {
          test[i] = Integer(64, i + 10, ALICE);
     }
     auto start = clock_start();
     uint64_t *a_share = context->gc->Y2A(test, length);
     cout << "Y2A time cost " << time_from(start) * 1.0 / 1000000 << endl;

     delete[] test;
     delete[] a_share;
}

void test_linear(SmcContext *context)
{
     Tensor input = torch::rand({2, 4}, *context->device);
     input.set_requires_grad(true);

     PlaintextLinear plinear(context, 4, 2);
     plinear.init();

     nn::LinearImpl linear(4, 2);
     linear.to(*context->device);
     linear.weight = plinear.getWeight().set_requires_grad(true);
     linear.bias = torch::zeros_like(linear.bias);

     cout << "weight" << endl
          << plinear.getWeight() << endl;

     Tensor res = linear.forward(input);
     (res / 2).sum().backward();
     cout << "grad" << endl
          << input.grad() << endl;

     Tensor ones = torch::ones({2, 2}, *context->device);
     Tensor sres = plinear.forward(input);
     Tensor sgrad = plinear.backward(ones);
     cout << "sgrad" << endl
          << sgrad / 2 << endl;
}

void test_slinear(SmcContext *context)
{
     int batch_size = 2, input_size = 3, output_size = 1;
     Tensor input = torch::randint(-10, 10, {1, batch_size, input_size}, *context->device) * context->ld;
     input = input.to(kLong);
     Stensor sinput(input, context);
     Tensor ones = torch::ones({batch_size, output_size}, *context->device).to(kLong) * context->ld;
     Stensor sones(ones / 2, context);

     TopSlinear slinear(context, batch_size, batch_size, input_size, output_size);
     slinear.revealMaskedX(input);
     Stensor sres = slinear.forward(0);
     slinear.backward(0, sones);

     cout << "sweight" << endl
          << slinear.getWeight().revealFloat() << endl;
     cout << "sres" << endl
          << sres.truncate_real(context->ldlen).revealFloat() << endl;
     cout << "weight grad" << endl
          << slinear.getGrad().truncate_real(context->ldlen).revealFloat() << endl;

     nn::LinearImpl linear(input_size, output_size);
     linear.to(*context->device);
     linear.weight = slinear.getWeight().revealFloat().set_requires_grad(true);
     linear.bias = torch::zeros_like(linear.bias);

     input = sinput.revealFloat()[0];
     input.set_requires_grad(true);
     Tensor res = linear.forward(input);
     (res).sum().backward();
     cout << "res" << endl
          << res << endl;
     cout << "weight grad" << endl
          << linear.weight.grad() << endl;
}

void test_sconv_signal(SmcContext *context)
{
     int in_channel = 32, out_channel = 64, kernel_size = 5;
     int batch_size = 128, image_size = 12, out_image_size = image_size - kernel_size + 1;

     Tensor input = torch::randint(-5, 5, {batch_size, in_channel, image_size, image_size}, *context->device);
     input = (input * context->ld).to(kLong);
     Tensor ones = torch::ones({batch_size, out_channel, out_image_size, out_image_size}, *context->device);
     ones = (ones * context->ld).to(kLong) / 2;
     Stensor sinput(input, context);
     Stensor sones(ones, context);

     SConv2D sconv1(context, in_channel, out_channel, kernel_size, batch_size, image_size);
     sconv1.genTriplets(3);

     for (int i = 0; i < 1; i++)
     {
          auto start = clock_start();
          Stensor sres1 = sconv1.forward(i, sinput);
          sres1 = sres1.truncate_real(context->ld);
          Stensor sgrad1 = sconv1.backward(i, sones);
          cout << "time cost " << time_from(start) / 1000000 << endl;

          //real
          nn::Conv2d conv1(nn::Conv2dOptions(in_channel, out_channel, kernel_size));
          conv1->to(*context->device);
          conv1->bias = torch::zeros_like(conv1->bias);
          conv1->weight = sconv1.getWeigth().revealFloat();
          conv1->weight.set_requires_grad(false);
          conv1->weight.set_requires_grad(true);

          input = sinput.revealFloat();
          input = input.set_requires_grad(true);
          Tensor res = conv1->forward(input);
          res.sum().backward();

          Tensor delta = sres1.revealFloat() - res;
          cout << "output delta" << endl
               << delta.max() << endl
               << delta.min() << endl;

          delta = sconv1.getWeigthGrad().revealFloat() / context->ld - conv1->weight.grad();
          cout << "weight1 grad delta" << endl
               << delta.max() << endl
               << delta.min() << endl;

          delta = sgrad1.revealFloat() / context->ld - input.grad();
          cout << "input grad delta" << endl
               << delta.max() << endl
               << delta.min() << endl;
     }
}

void test_sconv(SmcContext *context)
{
     int in_channel = 1, out_channel = 32, out_channel2 = 64, kernel_size = 5;
     int batch_size = 2, image_size = 28, out_image_size = image_size - kernel_size + 1;

     Tensor input = torch::randint(-3, 3, {batch_size, in_channel, image_size, image_size}, *context->device);
     input = (input * context->ld).to(kLong);
     Tensor ones = torch::ones({batch_size, out_channel2, out_image_size - kernel_size + 1, out_image_size - kernel_size + 1}, *context->device);
     ones = (ones * context->ld).to(kLong) / 2;
     Stensor sinput(input, context);
     Stensor sones(ones, context);

     SConv2D sconv1(context, in_channel, out_channel, kernel_size, batch_size, image_size);
     SConv2D sconv2(context, out_channel, out_channel2, kernel_size, batch_size, out_image_size);
     sconv1.genTriplets(3);
     sconv2.genTriplets(3);

     auto start = clock_start();
     Stensor sres1 = sconv1.forward(0, sinput);

     sres1 = sres1.truncate_real(context->ld);
     Stensor sres2 = sconv2.forward(0, sres1);

     Stensor sgrad2 = sconv2.backward(0, sones);
     sgrad2 = sgrad2.truncate_real(context->ld);

     Stensor sgrad1 = sconv1.backward(0, sgrad2);
     cout << "time cost " << time_from(start) / 1000000 << endl;

     nn::Conv2d conv1(nn::Conv2dOptions(in_channel, out_channel, kernel_size));
     conv1->to(*context->device);
     conv1->bias = torch::zeros_like(conv1->bias);
     conv1->weight = sconv1.getWeigth().revealFloat();
     conv1->weight.set_requires_grad(false);
     conv1->weight.set_requires_grad(true);
     nn::Conv2d conv2(nn::Conv2dOptions(out_channel, out_channel2, kernel_size));
     conv2->to(*context->device);
     conv2->bias = torch::zeros_like(conv2->bias);
     conv2->weight = sconv2.getWeigth().revealFloat();
     conv2->weight.set_requires_grad(false);
     conv2->weight.set_requires_grad(true);

     input = sinput.revealFloat();
     input = input.set_requires_grad(true);
     Tensor res1 = conv1->forward(input);
     Tensor res2 = conv2->forward(res1);
     res2 = res2 * sones.revealFloat();
     res2.sum().backward();

     Tensor delta = sres1.revealFloat() - res1;
     cout << "res1 delta" << endl
          << delta.max() << endl
          << delta.min() << endl;

     delta = sres2.revealFloat() / context->ld - res2;
     cout << "res2 delta" << endl
          << delta.max() << endl
          << delta.min() << endl;

     delta = sconv2.getWeigthGrad().revealFloat() / context->ld - conv2->weight.grad();
     cout << "weight2 grad delta" << endl
          << delta.max() << endl
          << delta.min() << endl;

     delta = sconv1.getWeigthGrad().revealFloat() / context->ld - conv1->weight.grad();
     cout << "weight1 grad delta" << endl
          << delta.max() << endl
          << delta.min() << endl;

     delta = sgrad1.revealFloat() / context->ld - input.grad();
     cout << "input grad delta" << endl
          << delta.max() << endl
          << delta.min() << endl;
}

void test_maxpool(SmcContext *context)
{
     Tensor input = torch::randint(-1000, 1000, {2, 3, 8, 8}, *context->device).to(kLong);
     Stensor sinput(input, context);
     SMaxPool2D pool(context);

     auto start = clock_start();
     Stensor spooled = pool.forward(sinput);
     cout << "forward time cost " << time_from(start) / 1000000 << endl;
     Tensor ones = torch::ones_like(spooled.data).to(kLong) * context->ld / 2;
     Stensor sgrad = pool.backward(Stensor(ones, context));
     cout << "backward time cost " << time_from(start) / 1000000 << endl;

     input = sinput.reveal().to(kFloat);
     input.set_requires_grad(true);
     Tensor pooled = max_pool2d(input, {2, 2});
     pooled.sum().backward();
     Tensor delta = spooled.reveal().to(kFloat) - pooled;
     cout << "pooled delta" << endl
          << delta.max() << endl
          << delta.min() << endl;

     delta = sgrad.revealFloat() - input.grad();
     cout << "input grad delta" << endl
          << delta.max() << endl
          << delta.min() << endl;
}

void test_compare(SmcContext *context)
{
     Tensor a = torch::randint(-500, 500, {2359296 / 4}).to(kLong);
     Tensor b = torch::randint(-500, 500, {2359296 / 4}).to(kLong);
     Tensor delta = a - b;
     long *data = delta.data<long>();

     auto start = clock_start();
     bool *signal = context->gc->gc_compare(data, delta.numel(), 16);
     signal = context->gc->gc_compare(data, delta.numel(), 16);
     signal = context->gc->gc_compare(data, delta.numel(), 16);
     cout << "time cost " << time_from(start) / 1000000 << endl;

     //cout << "delta" << endl
     //     << delta.clamp(-1, 0) << endl;
     for (int i = 0; i < delta.numel(); i++)
     {
          //printf("%02x \n", signal[i]);
     }
}

void test_cpu_speed(int role)
{
     Device device(kCUDA, role - 1);

     auto start = clock_start();
     long data[128 * 784];

     //Tensor a = torch::randint(0, 1l << 30, {128, 784}).to(kLong);
     Tensor a = torch::from_blob(data, {128, 784}, kLong).to(device);
     a = a.sum().cpu();
     //a = a.__lshift__(34).__or__(a);
     Tensor b = torch::from_blob(data, {784}, kLong).to(device);
     b = b.sum().cpu();
     //Tensor b = torch::randint(0, 1l << 30, {1, 784}).to(kLong);
     //b = b.__lshift__(34).__or__(b);

     //Tensor linear_res = torch::zeros({1, 128}, kLong);
     Tensor linear_res = torch::from_blob(data, {1, 784}, kLong).to(device);
     linear_res = linear_res.sum().cpu();
     for (int i = 0; i < 0; i++)
     {
          for (int j = 0; j < 1; j++)
          {
               Tensor res = a * b[j];
               res = res.sum(1);
               linear_res[j] = res;
          }
          linear_res.permute({1, 0});
     }
     cout << "cpu time cost " << time_from(start) / 1000000 << endl;
}

void test_truncate(SmcContext *context)
{
     short ld = 1 << 4;
     short Zp = 1 << 8;
     Tensor x = torch::randint(-8, 8, {1, 5}).to(kShort) * ld * ld;
     Tensor x1 = torch::randint(-65536, 65536, x.sizes()).to(kShort);
     Tensor x0 = x - x1;
     Tensor r = torch::randint(-65536, 65536, x.sizes()).to(kShort);
     x0 = x0 - r;
     x1 = x1 + r;
     x0 = (x0 % (1 << 12)).__rshift__(4);
     x1 = (x1 % (1 << 12)).__rshift__(4);
     Tensor x0_x1 = (x0 + x1).__and__(Zp - 1);
     x = (x / ld) % Zp;
     cout << x0_x1 << endl;
     cout << x << endl;
     Tensor delta = x0_x1 - x;
     cout << delta.max() << endl
          << delta.min() << endl
          << endl;
}

void slice_data()
{
     float acc_plaintext[120] = {0.098000, 0.236500, 0.503500, 0.647200, 0.642900, 0.654900, 0.715800, 0.728600, 0.669500, 0.699700, 0.704400, 0.727000, 0.731700, 0.732200, 0.744200, 0.737600, 0.765300, 0.760900, 0.763300, 0.769400, 0.751100, 0.763600, 0.770700, 0.758500, 0.774600, 0.783800, 0.787800, 0.782700, 0.788600, 0.794200, 0.800900, 0.801600, 0.806600, 0.810100, 0.800200, 0.802500, 0.804800, 0.817200, 0.812400, 0.814700, 0.812300, 0.812500, 0.808500, 0.816000, 0.814300, 0.811800, 0.809800, 0.809900, 0.811800, 0.820000, 0.819100, 0.822300, 0.824900, 0.823700, 0.827800, 0.828000, 0.828500, 0.828700, 0.831400, 0.839500, 0.838900, 0.829600, 0.833600, 0.835500, 0.832800, 0.829100, 0.827400, 0.825600, 0.830800, 0.832500, 0.831600, 0.834000, 0.836300, 0.836600, 0.839200, 0.842300, 0.838700, 0.842300, 0.840600, 0.840200, 0.840500, 0.842300, 0.843700, 0.846300, 0.844300, 0.845100, 0.843200, 0.840300, 0.840900, 0.843300, 0.846900, 0.845500, 0.846300, 0.846400, 0.843500, 0.839800, 0.837800, 0.835200, 0.841900, 0.847400, 0.850200, 0.850400, 0.851500, 0.846900, 0.845900, 0.850100, 0.853300, 0.851500, 0.850600, 0.852100, 0.854000, 0.856500, 0.855600, 0.855700, 0.856200, 0.860400, 0.858900, 0.859100, 0.860400, 0.861100};
     float acc_2[120] = {0.098000, 0.163900, 0.242600, 0.280800, 0.316800, 0.300500, 0.303900, 0.320900, 0.267400, 0.296100, 0.311400, 0.321300, 0.350700, 0.375600, 0.396500, 0.386400, 0.361500, 0.340700, 0.361600, 0.360400, 0.338700, 0.346900, 0.347100, 0.366800, 0.376600, 0.396900, 0.386300, 0.377700, 0.399400, 0.406600, 0.407500, 0.444000, 0.461700, 0.469900, 0.476800, 0.492900, 0.528300, 0.530500, 0.531600, 0.509200, 0.531700, 0.545200, 0.551500, 0.539900, 0.552800, 0.535200, 0.541700, 0.534800, 0.542900, 0.544500, 0.537700, 0.551800, 0.562800, 0.571100, 0.578000, 0.589600, 0.580900, 0.591200, 0.601500, 0.595800, 0.605100, 0.609000, 0.607000, 0.611600, 0.617200, 0.613600, 0.620800, 0.622800, 0.617200, 0.619800, 0.619400, 0.628100, 0.617700, 0.624600, 0.633200, 0.634200, 0.635800, 0.640800, 0.647400, 0.648100, 0.651800, 0.655900, 0.660600, 0.666700, 0.666900, 0.670200, 0.666200, 0.671800, 0.663600, 0.655500, 0.661300, 0.665300, 0.674100, 0.675000, 0.678000, 0.675100, 0.673700, 0.675400, 0.674100, 0.677000, 0.678500, 0.684200, 0.681300, 0.676700, 0.673500, 0.671400, 0.674100, 0.673700, 0.677200, 0.676200, 0.675400, 0.672700, 0.672700, 0.676600, 0.675600, 0.676400, 0.672800, 0.675500, 0.679400, 0.673500};
     float acc_4[120] = {0.098000, 0.247400, 0.434300, 0.485100, 0.484200, 0.479700, 0.517000, 0.540500, 0.536800, 0.534000, 0.618800, 0.619000, 0.620900, 0.649600, 0.632800, 0.636300, 0.679600, 0.684500, 0.697600, 0.718300, 0.709600, 0.717500, 0.719800, 0.695600, 0.703800, 0.712300, 0.714700, 0.712700, 0.714800, 0.715200, 0.724800, 0.730900, 0.747000, 0.740500, 0.739300, 0.738500, 0.740900, 0.747000, 0.736500, 0.745800, 0.733900, 0.737200, 0.738400, 0.742100, 0.746600, 0.748800, 0.753800, 0.749900, 0.759100, 0.765300, 0.765900, 0.765700, 0.770600, 0.771100, 0.771500, 0.776400, 0.776500, 0.776200, 0.784100, 0.787900, 0.784500, 0.783800, 0.787200, 0.783800, 0.778800, 0.776800, 0.775800, 0.776300, 0.778600, 0.780000, 0.781500, 0.778700, 0.781400, 0.783200, 0.784200, 0.786900, 0.787300, 0.789000, 0.790800, 0.791700, 0.792900, 0.792900, 0.797100, 0.804300, 0.801000, 0.803900, 0.801400, 0.797600, 0.800500, 0.802300, 0.805200, 0.804800, 0.803000, 0.804600, 0.799000, 0.793600, 0.795200, 0.792400, 0.800900, 0.803600, 0.804500, 0.807400, 0.808300, 0.804700, 0.805500, 0.809500, 0.812300, 0.812800, 0.811600, 0.811400, 0.813100, 0.819100, 0.817200, 0.819000, 0.820600, 0.823800, 0.825000, 0.827700, 0.827700, 0.827500};
     float acc_6[120] = {0.098000, 0.319300, 0.426600, 0.572700, 0.572700, 0.615200, 0.665100, 0.658200, 0.606600, 0.618500, 0.664300, 0.655300, 0.652500, 0.678500, 0.686000, 0.681700, 0.701500, 0.696900, 0.702800, 0.706700, 0.699400, 0.702000, 0.711100, 0.693000, 0.705100, 0.717900, 0.720800, 0.723500, 0.732900, 0.731400, 0.741000, 0.734900, 0.748400, 0.742300, 0.735100, 0.730500, 0.734400, 0.755800, 0.754200, 0.756800, 0.753300, 0.759600, 0.754800, 0.767000, 0.774100, 0.775000, 0.773500, 0.772000, 0.774200, 0.784100, 0.783300, 0.786200, 0.789600, 0.787700, 0.787600, 0.790600, 0.796700, 0.797400, 0.800500, 0.807800, 0.804600, 0.802600, 0.806700, 0.804700, 0.799400, 0.797100, 0.796700, 0.796200, 0.798800, 0.798000, 0.797200, 0.800700, 0.807500, 0.808800, 0.810600, 0.811400, 0.811700, 0.815600, 0.813700, 0.814200, 0.814500, 0.815500, 0.818900, 0.819800, 0.818800, 0.819400, 0.815200, 0.814300, 0.816200, 0.820700, 0.824600, 0.823900, 0.821600, 0.824500, 0.822300, 0.818600, 0.818000, 0.816200, 0.821400, 0.823800, 0.825200, 0.824400, 0.826100, 0.822600, 0.823400, 0.823300, 0.830800, 0.830300, 0.828300, 0.828000, 0.830300, 0.835000, 0.831700, 0.831200, 0.833100, 0.837000, 0.837100, 0.837000, 0.839100, 0.837700};
     float loss_plaintext[120] = {2.302585, 2.227588, 2.165033, 2.100985, 2.089835, 2.007065, 1.935144, 1.921456, 1.925906, 1.828697, 1.772637, 1.752091, 1.711852, 1.592382, 1.577247, 1.562886, 1.447831, 1.484109, 1.491332, 1.440306, 1.366615, 1.428875, 1.276789, 1.394726, 1.345883, 1.337458, 1.291580, 1.302393, 1.181366, 1.248957, 1.168141, 1.198882, 1.171021, 1.148654, 1.188237, 0.996679, 1.096331, 1.201087, 1.069767, 1.051473, 1.041948, 1.020482, 0.910423, 1.001555, 1.008645, 1.047110, 0.936457, 0.767311, 0.928616, 0.925926, 0.924599, 0.765274, 0.956609, 1.059260, 0.957582, 0.944240, 1.172847, 1.053711, 1.021615, 0.939187, 1.003067, 0.903380, 0.879657, 0.843305, 1.018885, 0.909075, 0.740605, 0.897968, 0.990552, 1.011384, 0.706129, 0.822745, 0.850924, 0.816421, 0.816767, 0.695028, 0.775545, 0.702026, 0.755645, 0.759393, 0.681974, 0.661429, 0.652910, 0.777800, 0.704756, 0.682290, 0.652506, 0.748466, 0.685951, 0.664238, 0.900222, 0.868059, 0.758623, 0.729511, 0.682890, 0.745188, 0.793092, 0.803975, 0.883095, 0.791758, 0.710881, 0.882237, 0.855856, 0.725343, 0.711627, 0.562652, 0.696162, 0.768284, 0.785975, 0.860526, 0.761591, 0.809297, 0.828147, 0.805979, 1.013608, 0.983886, 0.633251, 0.572019, 0.765436, 0.626872};
     float loss_2[120] = {2.302585, 2.297829, 2.260561, 2.195853, 2.187668, 2.148747, 2.062352, 2.073899, 2.089571, 2.090342, 2.005138, 1.991365, 1.955033, 1.860591, 1.909913, 1.953755, 1.818911, 2.016263, 1.926361, 1.872561, 1.727101, 1.932230, 1.784951, 1.717015, 1.820748, 1.714014, 1.809348, 1.730472, 1.664368, 1.737446, 1.705965, 1.671018, 1.622061, 1.550834, 1.614498, 1.508025, 1.589048, 1.558986, 1.324652, 1.402241, 1.454496, 1.305084, 1.331673, 1.448897, 1.452536, 1.512521, 1.318937, 1.292496, 1.419120, 1.340017, 1.308016, 1.159805, 1.398121, 1.470045, 1.324116, 1.179616, 1.498851, 1.335286, 1.411874, 1.359017, 1.359810, 1.201679, 1.261673, 1.152169, 1.331991, 1.249078, 1.079942, 1.139751, 1.276799, 1.252861, 1.020976, 1.136128, 1.246881, 1.093928, 1.176940, 0.924799, 0.995605, 0.881663, 0.988816, 1.101790, 0.997922, 1.012511, 0.943798, 0.997863, 0.991806, 0.886568, 0.818505, 0.908224, 0.933417, 0.973479, 1.222446, 1.304775, 1.097374, 1.025949, 1.026726, 1.010065, 1.171250, 1.120538, 1.094454, 1.099130, 0.925736, 1.238909, 1.204760, 1.028599, 0.944497, 0.739835, 0.991918, 1.066092, 1.120752, 1.164568, 1.122046, 1.085716, 1.270798, 1.158236, 1.373375, 1.274564, 0.897790, 0.929702, 1.111947, 1.022322};
     float loss_4[120] = {2.302585, 2.248473, 2.159132, 2.090526, 2.085083, 2.026041, 1.952047, 1.965679, 1.915941, 1.857631, 1.736277, 1.741485, 1.699202, 1.576720, 1.585526, 1.561131, 1.436364, 1.506189, 1.507630, 1.406330, 1.324858, 1.388776, 1.234773, 1.317938, 1.271632, 1.314334, 1.238984, 1.276832, 1.128358, 1.230426, 1.109793, 1.196961, 1.133736, 1.067852, 1.086985, 0.950130, 1.063753, 1.192806, 1.081856, 0.984117, 0.993091, 0.953321, 0.882055, 0.964988, 0.959902, 0.990368, 0.853773, 0.700512, 0.871252, 0.862625, 0.850830, 0.668550, 0.896192, 0.987332, 0.912305, 0.885506, 1.103956, 1.020666, 0.958627, 0.845252, 0.907672, 0.860808, 0.849252, 0.764154, 0.965585, 0.845363, 0.680571, 0.863433, 1.030391, 0.994756, 0.663254, 0.807863, 0.838256, 0.774391, 0.794407, 0.630871, 0.774802, 0.602037, 0.734344, 0.674246, 0.607423, 0.649301, 0.580185, 0.679420, 0.646555, 0.600311, 0.588082, 0.690060, 0.627536, 0.650987, 0.984261, 0.891643, 0.695228, 0.692944, 0.680077, 0.737914, 0.746048, 0.772956, 0.867824, 0.776125, 0.669103, 0.844010, 0.863302, 0.693299, 0.685903, 0.519377, 0.657745, 0.686359, 0.714853, 0.859455, 0.753380, 0.798861, 0.809974, 0.759669, 1.027715, 1.006923, 0.561512, 0.472267, 0.698346, 0.552263};
     float loss_6[120] = {2.302585, 2.229166, 2.174357, 2.098106, 2.099025, 2.001824, 1.932076, 1.911401, 1.916763, 1.832423, 1.728897, 1.748469, 1.697882, 1.559978, 1.557189, 1.549442, 1.392078, 1.466594, 1.453224, 1.384073, 1.301241, 1.375309, 1.192249, 1.356467, 1.285579, 1.261576, 1.211888, 1.229815, 1.096082, 1.149862, 1.062161, 1.155533, 1.085232, 1.024686, 1.080262, 0.898731, 1.059052, 1.115461, 0.971510, 0.927844, 0.947866, 0.911209, 0.852060, 0.938827, 0.933532, 0.927237, 0.849289, 0.672357, 0.853174, 0.830435, 0.847996, 0.653796, 0.834064, 0.962143, 0.887588, 0.838979, 1.090526, 0.966307, 0.971624, 0.828202, 0.893056, 0.856390, 0.825520, 0.724881, 0.942328, 0.831886, 0.647211, 0.859557, 0.930058, 0.949927, 0.613388, 0.734107, 0.762750, 0.742249, 0.725243, 0.595037, 0.717259, 0.618896, 0.657106, 0.675829, 0.593115, 0.576790, 0.583009, 0.665871, 0.613082, 0.586261, 0.541672, 0.679892, 0.618068, 0.573676, 0.876676, 0.814621, 0.674630, 0.633214, 0.591646, 0.679239, 0.711678, 0.733522, 0.857472, 0.719524, 0.632018, 0.835944, 0.805800, 0.650766, 0.646576, 0.485128, 0.602361, 0.703932, 0.716441, 0.823108, 0.709678, 0.760950, 0.753567, 0.712498, 1.006727, 0.946460, 0.544751, 0.471771, 0.706688, 0.539438};

     cout << "------------ acc_plaintext ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", acc_plaintext[i]);
     cout << endl;

     cout << "------------ acc_2 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", acc_2[i]);
     cout << endl;

     cout << "------------ acc_4 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", acc_4[i]);
     cout << endl;

     cout << "------------ acc_6 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", acc_6[i]);
     cout << endl;

     cout << "------------ loss_plaintext ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", loss_plaintext[i]);
     cout << endl;

     cout << "------------ loss_2 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", loss_2[i]);
     cout << endl;

     cout << "------------ loss_4 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", loss_4[i]);
     cout << endl;

     cout << "------------ loss_6 ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%f,", loss_6[i]);
     cout << endl;

     cout << "------------ index ---------------" << endl;
     for (int i = 0; i < 120; i++)
          if (i % 5 == 0)
               printf("%d,", i);
     cout << endl;
}

int main(int argc, char **argv)
{
     //NoGradGuard guard;
     /***** init context *****/
     SmcContext *context = new SmcContext(atoi(argv[1]), argv[2], atoi(argv[3]));
     context->setLD(4);

     for (int i = 0; i < 1; i++)
          test_sigmod(context);

     return 0;
}
