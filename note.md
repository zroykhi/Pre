batch normalization is very important, without it, prediction outcomes sometimes are all the same
if the leatning rate is too high, the loss can sometimes appears to be nan

for rnn model, the parameters below work
lr = 1e-5,
hidden_size=32,
num_layers=32,
CAMERA_RESOLUTION = (486, 270)
FACTOR = 3
batch size is 4

difference of LSTM between return hidden state and not return?
how to add batch normalization to rnn

LSTM模型要设置train.py中loss.backward(retain_graph=True)，并取消转化h_state每一步time step之后的格式，
# todo
prendre plusieurs images pour entrainer. bouger le bateau, predire les parameters du bateau quelque seconds plus tard. tester different interval de frame
donnees pour entrainer et pour tester sont separer.

blender error: RuntimeError: Error: Cannot write a single file with an animation format selected
solution: change render format to png or jpg

modify learning rate and batch size sometimes can change a lot

time gap should according to the boat's speed

不能直接使用rnn，先经过cnn对图片的特征读取，再传入rnn中

修改time_gap,

when using cnn_rnn model, can not use train_test_split !!!

change the wave, it is too chaotic now

courve: prediction and origin
courve: loss and time gap
cnn prendre 2 images comme entre

change the local axis direction by simply changing the value of z rotation in the menu

# if you use a trained net's parameters
class newNet(self, net)
for para in net.parameters():
  para.grad_required = False # disable gradient

difference between return hidden state and not return ?
why random split data for lstm works better?

implemente the data augmentation during the training

保证utils.py和test.py中的两张图片的间隔是一样的

# test的结果不好是因为roll和pitch的位置改变了,在blender模型里面，预测的船和原船的local坐标系xy互换了！！！
> 纠正方法是，改变blender model里面的plane1的local坐标系的角度，使得boat1相应的变换，最终使
> 得boat和boat1的local坐标系方向一致

data load 是不按顺序的，需要修改。

utils.py中的labels的顺序要随着keys排序而排序！！！

DataLoader的shuffle改为true，训练才有效果

loadlabesl use sequential load. and change the gap of two pictures(3) seems works? non
though validation loss is big, the predictions maybe work as well

sudo python3 -m Pre.test -f Pre/3dmodel/testData719/ -w Pre/results/CNN_LSTM_model_25_tmp.pth -m CNN_LSTM
sudo ppython3 -m Pre.train -f Pre/3model/render706/ --model_type CNN_LSTM -bs 8

# results record:
1. loadLabels: sequence; JsonDataset: two images cnn sequence, predict original data works;6 images gap; predict testData works a little(in some place).
2. loadLabels: sequence; JsonDataset: two images cnn not sequence, predict original data works moins bon;6 images gap; predict testData works moins bon(especially pitch).
3. loadLabels: not sequence; JsonDataset: two images cnn not sequence, predict original data **works best**;6 images gap; predict testData works so-so.
could try half for train half for test
4. loadLabels: not sequence; JsonDataset: two images cnn sequence, predict original data works moins bon;6 images gap; predict testData works moins bon(especially pitch).

Lstm
5. loadLabels: sequence; JsonDataset: sequence; predict original data roll good, pitch bad; predict testData predict testData roll pretty good, pitch not good; val_loss not converge
6. loadLabels: not sequence; JsonDataset: sequence; predict original data  roll good, pitch bad; predict testData roll pretty good, pitch not good;
7. loadLabels: sequence; JsonDataset: not sequence; predict original data same with above; predict testData roll very good, pitch bad;
8. loadLabels: not sequence; JsonDataset: not sequence; predict original data; predict testData ;

9. one image cnn, prediction of original data is good, bad for prediction of testData
train epochs less

# data needed:
1. one image only cnn
2. one image only cnn_lstm
3. two images cnn

weight_decay up, delete first F.dropout, more fiters; optimizer?l2, use only cnn no Linear? activation function?
restnet, random_trans, use different angle together to train!! more pics 3/4 together to train!!
loss functioin, suitable? every y take into account? batch size, don't use imread!!!
make sure 'num_workers' in train.py doesn't exceed your cpu core num.

适当增大weight decay可以让pitch的prediction更准确
