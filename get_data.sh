# download and extract the CIFAR-10 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

mkdir data
current_dir=$(pwd)
mv cifar-10-python.tar.gz "$current_dir/data/"
cd "$current_dir/data/"
tar xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
