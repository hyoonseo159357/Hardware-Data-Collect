nvidia-smi , dstat 을 수집함

- aws configure 로 리전 맞춰주고

cuda11.4 /
tensorflow 2.7.0 /
python 3.7 /
N.virginia / Deep Learning AMI GPU CUDA 11.4.1 (Ubuntu 18.04) 20211204 AMI)

- 로컬에서 파일 다운로드받고, 파일안에 pem 키넣고
- sh startCLI_Oregon.sh g4dn.xlarge

다깔고 ssh 접속되면
- cd Hardware-Data-Collect
- sudo bash ./Run_nvidiasmi_dstat.sh (gpu+cpu둘다수집)
- sudo bash ./Run_nvidiasmi.sh (gpu만수집)
- sudo bash ./Run_DCGMI.sh (dcgmi로 gpu수집)

