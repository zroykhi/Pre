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
