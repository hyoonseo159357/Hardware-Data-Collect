tmux new -s dstattmux
sudo dstat --output ~/Hardware-Data/dstat-log.csv -cdnpmrt
tmux detach
tmux new -s nvidiasmitmux
nvidia-smi -lms 110 -f ./Data.csv --format=csv --query-gpu=timestamp,temperature.gpu,utilization.memory,utilization.gpu,power.draw,clocks.current.sm
tmux detach
tmux new -s maintmux


# 'LeNet5'
python3.7 workload.py --model 'LeNet5' --dataset 32 --batch_size 16    
# VGGSmall
python3.7 workload.py --model 'VGGSmall' --dataset 32 --batch_size 16    
# VGG11
python3.7 workload.py --model 'VGG11' --dataset 32 --batch_size 16    
# VGG13
python3.7 workload.py --model 'VGG13' --dataset 32 --batch_size 16    
# VGG16
python3.7 workload.py --model 'VGG16' --dataset 32 --batch_size 16    
# VGG19
python3.7 workload.py --model 'VGG16' --dataset 32 --batch_size 16    
# ResNetSmall
python3.7 workload.py --model 'ResNetSmall' --dataset 32 --batch_size 16    
# ResNet18
python3.7 workload.py --model 'ResNet18' --dataset 32 --batch_size 16    
# ResNet34
python3.7 workload.py --model 'ResNet34' --dataset 32 --batch_size 16    
# MNIST_CNN
python3.7 workload.py --model 'MNIST_CNN' --dataset 32 --batch_size 16    
# CIFAR10_CNN
python3.7 workload.py --model 'CIFAR10_CNN' --dataset 32 --batch_size 16    
# FLOWER_CNN
python3.7 workload.py --model 'FLOWER_CNN' --dataset 32 --batch_size 16    
# AlexNet
python3.7 workload.py --model 'AlexNet' --dataset 32 --batch_size 16    
# InceptionV3
python3.7 workload.py --model 'InceptionV3' --dataset 32 --batch_size 16    
# InceptionResNetV2
python3.7 workload.py --model 'InceptionResNetV2' --dataset 32 --batch_size 16    
# Xception
python3.7 workload.py --model 'Xception' --dataset 32 --batch_size 16    
# EfficientNetB0
python3.7 workload.py --model 'EfficientNetB0' --dataset 32 --batch_size 16    
# MobileNetV2
python3.7 workload.py --model 'MobileNetV2' --dataset 32 --batch_size 16    
# ResNet50
python3.7 workload.py --model 'ResNet50' --dataset 32--batch_size 16
